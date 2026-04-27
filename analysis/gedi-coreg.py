#!/usr/bin/env python
"""
gedi_dtm_coreg.py

Co-register GEDI L2A footprints to an airborne lidar DTM using ASP pc_align,
operating per-orbit. Saves aligned footprints and co-registration shift vectors.

I used ChatGSFC help to port my notebook code into this script, so if you see any weirdness please let me know at shashank.bhushan@nasa.gov


Usage:
    python gedi_dtm_coreg.py \
        --dtm /path/to/dtm.tif \
        --gedi /path/to/gedi.gpkg \
        --outdir /path/to/output \
        --asp-dir /path/to/StereoPipeline/bin \
        --max-displacement 40 \
        --alignment-method point-to-plane \
        --min-points 100 \
        --verbose
"""

import os
import sys
import glob
import argparse
import json
from datetime import datetime, timezone
from shutil import which
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pcd_altimetry_coreg_plot import plot_gedi_coreg_results


# =============================================================================
# Parsing pc_align output
# =============================================================================

def parse_pc_align_log(log_path):
    """
    Parse pc_align log file for translation, rotation, scale, and total shift.

    Parameters
    ----------
    log_path : str
        Path to pc_align log file.

    Returns
    -------
    dict with keys:
        total_displacement : float (meters)
        translation_ned : tuple (north, east, down) in meters
        translation_enu : tuple (east, north, up) in meters
        euler_angles_ned : tuple (north, east, down) in degrees
        scale : float
    """
    with open(log_path, "r") as f:
        content = f.readlines()

    # Total displacement
    substr = (
        "Maximum displacement of points between the source cloud with any "
        "initial transform applied to it and the source cloud after alignment "
        "to the reference"
    )
    match = [line for line in content if substr in line]
    total_displacement = float(match[0].split(":", 15)[-1].split("m")[0])

    # Translation vector (North-East-Down)
    substr = "Translation vector (North-East-Down, meters):"
    line = [l for l in content if substr in l][0]
    vec = line.split("Vector3")[1]
    north = float(vec.split("(")[1].split(",")[0])
    east = float(vec.split(",")[1])
    down = float(vec.split(",")[2].split(")")[0])

    # Euler angles (North-East-Down)
    substr = " Euler angles (North-East-Down, degrees)"
    line = [l for l in content if substr in l][0]
    vec = line.split("Vector3")[1]
    north_angle = float(vec.split("(")[1].split(",")[0])
    east_angle = float(vec.split(",")[1])
    down_angle = float(vec.split(",")[2].split(")")[0])

    # Scale
    substr = "Transform scale"
    line = [l for l in content if substr in l][0]
    scale = float(line.split("= ")[1].split("\n")[0]) + 1

    return {
        "total_displacement": total_displacement,
        "translation_ned": (north, east, down),
        "translation_enu": (east, north, -1 * down),
        "euler_angles_ned": (north_angle, east_angle, down_angle),
        "scale": scale,
    }


# =============================================================================
# Apply shift to GeoDataFrame
# =============================================================================

def apply_shift(gdf, dx, dy, dz, elevation_col="elevation_lm"):
    """
    Apply a rigid translation to GEDI footprint positions and elevations.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GEDI footprints with geometry and elevation column.
    dx, dy, dz : float
        Easting, northing, and vertical shifts in meters.
    elevation_col : str
        Name of the elevation column.

    Returns
    -------
    GeoDataFrame
        Shifted footprints with updated geometry and elevation.
    """
    crs = gdf.estimate_utm_crs()
    out = gdf.copy()
    out[elevation_col] = gdf[elevation_col] + dz
    out["easting"] = gdf.geometry.x + dx
    out["northing"] = gdf.geometry.y + dy
    geometry = [Point(x, y) for x, y in zip(out["easting"], out["northing"])]
    out = gpd.GeoDataFrame(out.drop(columns=["geometry"]), geometry=geometry, crs=crs)
    return out


# =============================================================================
# Core co-registration function
# =============================================================================

def coreg_gedi_to_dtm(
    dtm_path,
    gedi_gpkg_path,
    outdir,
    asp_bin_dir=None,
    alignment_method="point-to-plane",
    max_displacement=40,
    min_points=100,
    elevation_col="elevation_lm",
    verbose=False,
):
    """
    Co-register GEDI footprints to a DTM using ASP pc_align, per orbit.

    Parameters
    ----------
    dtm_path : str
        Path to reference DTM GeoTIFF (ideally Gaussian-blurred to ~sigma 5.5 m).
    gedi_gpkg_path : str
        Path to GEDI footprints GeoPackage with columns:
        - elevation_lm (or as specified)
        - orbit
        - geometry (Point, in projected CRS matching DTM)
    outdir : str
        Output directory for alignment results.
    asp_bin_dir : str, optional
        Path to ASP bin directory. If None, assumes ASP tools are on PATH.
    alignment_method : str
        pc_align alignment method ('point-to-plane' or 'point-to-point').
    max_displacement : float
        Maximum expected displacement in meters.
    min_points : int
        Minimum footprints per orbit to attempt alignment.
    elevation_col : str
        Column name for GEDI terrain elevation.
    verbose : bool
        Print pc_align stdout/stderr.

    Returns
    -------
    aligned_gdf : GeoDataFrame
        All aligned GEDI footprints concatenated.
    shift_gdf : GeoDataFrame
        One row per orbit with shift vectors and metadata (for GeoJSON export).
    """
    import subprocess
    import rioxarray as rxr

    os.makedirs(outdir, exist_ok=True)

    # Resolve ASP tool paths
    def _find_tool(name):
        if asp_bin_dir:
            path = os.path.join(asp_bin_dir, name)
            if os.path.isfile(path):
                return path
        found = which(name)
        if found:
            return found
        raise FileNotFoundError(
            f"Cannot find '{name}'. Set --asp-dir or add ASP to PATH."
        )

    pc_align_bin = _find_tool("pc_align")
    geodiff_bin = _find_tool("geodiff")

    # Load DTM metadata for CRS/projection
    da_dtm = rxr.open_rasterio(dtm_path, masked=True).squeeze()
    dtm_crs = da_dtm.rio.crs

    # Write WKT for csv-srs
    wkt_path = os.path.join(outdir, "dtm_crs.wkt")
    with open(wkt_path, "w") as f:
        f.write(dtm_crs.to_wkt())

    csv_format = "1:easting,2:northing,3:height_above_datum"

    # Load GEDI data
    gf_gedi = gpd.read_file(gedi_gpkg_path)

    # Ensure projected CRS matches DTM
    if gf_gedi.crs != dtm_crs:
        print(f"Reprojecting GEDI from {gf_gedi.crs} to {dtm_crs}")
        gf_gedi = gf_gedi.to_crs(dtm_crs)

    gf_gedi["easting"] = gf_gedi.geometry.x
    gf_gedi["northing"] = gf_gedi.geometry.y

    orbits = np.unique(gf_gedi["orbit"].values)
    print(f"Found {len(orbits)} orbits, {len(gf_gedi)} total footprints")

    aligned_gdf_list = []
    shift_records = []

    for orbit in orbits:
        gedi_orbit = gf_gedi[gf_gedi["orbit"] == orbit].copy()
        num_points = len(gedi_orbit)

        # Compute orbit centroid for GeoJSON shift vector
        centroid_x = gedi_orbit.geometry.x.mean()
        centroid_y = gedi_orbit.geometry.y.mean()

        if num_points < min_points:
            print(
                f"[orbit {orbit}] Skipping: {num_points} points < "
                f"minimum {min_points}"
            )
            shift_records.append(
                _make_shift_record(
                    orbit, num_points, centroid_x, centroid_y,
                    status="skipped_insufficient_points",
                )
            )
            continue

        print(f"[orbit {orbit}] Aligning {num_points} footprints...")

        # Write orbit CSV for pc_align
        orbit_csv = os.path.join(outdir, f"gedi_orbit_{orbit}.csv")
        gedi_orbit[["easting", "northing", elevation_col]].to_csv(
            orbit_csv, index=False
        )

        # Run pc_align
        align_prefix = os.path.join(
            outdir, f"align_orbit_{orbit}", "gedi2dtm"
        )
        os.makedirs(os.path.dirname(align_prefix), exist_ok=True)

        cmd = [
            pc_align_bin,
            "--compute-translation-only",
            "--highest-accuracy",
            "--csv-format", csv_format,
            "--csv-srs", wkt_path,
            "--save-transformed-source-points",
            "--alignment-method", alignment_method,
            "--max-displacement", str(max_displacement),
            dtm_path,
            orbit_csv,
            "-o", align_prefix,
        ]

        result = _run_command(cmd, verbose=verbose)

        if result.returncode != 0:
            print(f"[orbit {orbit}] pc_align failed (returncode={result.returncode})")
            shift_records.append(
                _make_shift_record(
                    orbit, num_points, centroid_x, centroid_y,
                    status="pc_align_failed",
                )
            )
            continue

        # Parse alignment results
        log_files = sorted(glob.glob(f"{align_prefix}*log*pc_align*.txt"))
        if not log_files:
            print(f"[orbit {orbit}] No log file found")
            shift_records.append(
                _make_shift_record(
                    orbit, num_points, centroid_x, centroid_y,
                    status="no_log_found",
                )
            )
            continue

        params = parse_pc_align_log(log_files[-1])
        dx, dy, dz = params["translation_enu"]

        print(
            f"[orbit {orbit}] dx={dx:.2f} m, dy={dy:.2f} m, "
            f"dz={dz:.2f} m, total={params['total_displacement']:.2f} m"
        )

        # Apply shift
        aligned_orbit = apply_shift(gedi_orbit, dx, dy, dz, elevation_col)
        aligned_gdf_list.append(aligned_orbit)

        # Record shift vector
        shift_records.append(
            _make_shift_record(
                orbit, num_points, centroid_x, centroid_y,
                status="success",
                dx=dx, dy=dy, dz=dz,
                total_displacement=params["total_displacement"],
                north_angle=params["euler_angles_ned"][0],
                east_angle=params["euler_angles_ned"][1],
                down_angle=params["euler_angles_ned"][2],
                scale=params["scale"],
            )
        )

    # Concatenate aligned orbits
    if aligned_gdf_list:
        aligned_gdf = pd.concat(aligned_gdf_list, ignore_index=True)
        aligned_gdf = gpd.GeoDataFrame(aligned_gdf, crs=aligned_gdf_list[0].crs)
    else:
        print("WARNING: No orbits were successfully aligned.")
        aligned_gdf = gpd.GeoDataFrame(columns=gf_gedi.columns, crs=dtm_crs)

    # Build shift vector GeoDataFrame
    shift_gdf = _build_shift_gdf(shift_records, dtm_crs)

    return aligned_gdf, shift_gdf


# =============================================================================
# Shift vector record helpers
# =============================================================================

def _make_shift_record(
    orbit, num_points, centroid_x, centroid_y,
    status="unknown",
    dx=np.nan, dy=np.nan, dz=np.nan,
    total_displacement=np.nan,
    north_angle=np.nan, east_angle=np.nan, down_angle=np.nan,
    scale=np.nan,
):
    """Create a dict representing one orbit's co-registration result."""
    return {
        "orbit": int(orbit),
        "num_points": int(num_points),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "status": status,
        "dx_east_m": dx,
        "dy_north_m": dy,
        "dz_up_m": dz,
        "total_displacement_m": total_displacement,
        "euler_north_deg": north_angle,
        "euler_east_deg": east_angle,
        "euler_down_deg": down_angle,
        "scale": scale,
    }


def _build_shift_gdf(records, crs):
    """
    Build a GeoDataFrame of shift vectors, one row per orbit.
    Geometry is the orbit centroid in projected CRS.
    """
    df = pd.DataFrame(records)
    geometry = [Point(r["centroid_x"], r["centroid_y"]) for r in records]
    gdf = gpd.GeoDataFrame(
        df.drop(columns=["centroid_x", "centroid_y"]),
        geometry=geometry,
        crs=crs,
    )
    return gdf


# =============================================================================
# Geodiff wrapper
# =============================================================================

def compute_geodiff(csv_path, dtm_path, wkt_path, output_prefix, geodiff_bin,
                    verbose=False):
    """Run ASP geodiff between a CSV point cloud and a DTM."""
    csv_format = "1:easting,2:northing,3:height_above_datum"
    cmd = [
        geodiff_bin,
        csv_path, dtm_path,
        "--csv-format", csv_format,
        "--csv-srs", wkt_path,
        "-o", output_prefix,
    ]
    _run_command(cmd, verbose=verbose)
    diff_files = glob.glob(f"{output_prefix}-diff.csv")
    if diff_files:
        return diff_files[0]
    return None


# =============================================================================
# Subprocess runner
# =============================================================================

def _run_command(cmd, verbose=False):
    """Run a shell command, optionally printing output."""
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
    )
    if not verbose and result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}" if result.stderr else "")
    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Co-register GEDI footprints to a DTM using ASP pc_align.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dtm", required=True,
        help="Path to reference DTM GeoTIFF (ideally Gaussian-blurred).",
    )
    parser.add_argument(
        "--gedi", required=True,
        help="Path to GEDI footprints GeoPackage (projected CRS, with 'orbit' "
             "and 'elevation_lm' columns).",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--asp-dir", default=None,
        help="Path to ASP bin directory. If not set, tools must be on PATH.",
    )
    parser.add_argument(
        "--alignment-method", default="point-to-plane",
        choices=["point-to-plane", "point-to-point"],
        help="pc_align alignment method (default: point-to-plane).",
    )
    parser.add_argument(
        "--max-displacement", type=float, default=40,
        help="Maximum expected displacement in meters (default: 40).",
    )
    parser.add_argument(
        "--min-points", type=int, default=100,
        help="Minimum footprints per orbit to attempt alignment (default: 100).",
    )
    parser.add_argument(
        "--elevation-col", default="elevation_lm",
        help="GEDI elevation column name (default: elevation_lm).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print pc_align stdout/stderr.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GEDI-DTM Co-registration")
    print(f"  DTM:    {args.dtm}")
    print(f"  GEDI:   {args.gedi}")
    print(f"  Output: {args.outdir}")
    print("=" * 60)

    # Run co-registration
    aligned_gdf, shift_gdf = coreg_gedi_to_dtm(
        dtm_path=args.dtm,
        gedi_gpkg_path=args.gedi,
        outdir=args.outdir,
        asp_bin_dir=args.asp_dir,
        alignment_method=args.alignment_method,
        max_displacement=args.max_displacement,
        min_points=args.min_points,
        elevation_col=args.elevation_col,
        verbose=args.verbose,
    )

    # Save aligned footprints
    aligned_gpkg = os.path.join(args.outdir, "gedi_aligned.gpkg")
    
    aligned_gdf.to_file(aligned_gpkg, driver="GPKG")

    
    print(f"Aligned footprints saved: {aligned_gpkg}")

    # Save shift vectors as GeoJSON (reproject to EPSG:4326 for portability)
    shift_geojson = os.path.join(args.outdir, "coreg_shift_vectors.geojson")
    shift_gdf_4326 = shift_gdf.to_crs(epsg=4326)
    shift_gdf_4326.to_file(shift_geojson, driver="GeoJSON")
    print(f"Shift vectors saved: {shift_geojson}")

    # Print summary
    print("\n" + "=" * 60)
    print("Co-registration Summary")
    print("=" * 60)
    success = shift_gdf[shift_gdf["status"] == "success"]
    skipped = shift_gdf[shift_gdf["status"] != "success"]
    print(f"  Orbits aligned:  {len(success)}/{len(shift_gdf)}")
    print(f"  Orbits skipped:  {len(skipped)}")
    if len(success) > 0:
        print(f"\n  {'Orbit':>10}  {'dx(E)':>8}  {'dy(N)':>8}  {'dz(U)':>8}  "
              f"{'Total':>8}  {'N pts':>6}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
        for _, row in success.iterrows():
            print(
                f"  {row['orbit']:>10}  "
                f"{row['dx_east_m']:>8.2f}  "
                f"{row['dy_north_m']:>8.2f}  "
                f"{row['dz_up_m']:>8.2f}  "
                f"{row['total_displacement_m']:>8.2f}  "
                f"{row['num_points']:>6}"
            )
        print(f"\n  Mean shift: "
              f"dx={success['dx_east_m'].mean():.2f} m, "
              f"dy={success['dy_north_m'].mean():.2f} m, "
              f"dz={success['dz_up_m'].mean():.2f} m")
    ## Geodiff and plotting component
    original_csv_fn = os.path.join(args.outdir, "non_aligned.csv")
    aligned_csv_fn = os.path.join(args.outdir, "aligned.csv")
    gpd.read_file(args.gedi)[["easting", "northing", args.elevation_col]].to_csv(original_csv_fn,index=False, header=False)
    aligned_csv  = aligned_gdf[["easting", "northing", args.elevation_col]].to_csv(aligned_csv_fn,index=False, header=False)
    original_geodiff_fn = compute_geodiff(
        csv_path=original_csv_fn,
        dtm_path=args.dtm,
        wkt_path=os.path.join(args.outdir, "dtm_crs.wkt"),
        output_prefix=os.path.join(args.outdir, "original-geodiff"),
        geodiff_bin=which("geodiff") if args.asp_dir is None else os.path.join(args.asp_dir, "geodiff"),
        verbose=args.verbose,
    )

    aligned_geodiff_fn = compute_geodiff(
        csv_path=aligned_csv_fn,
        dtm_path=args.dtm,
        wkt_path=os.path.join(args.outdir, "dtm_crs.wkt"),
        output_prefix=os.path.join(args.outdir, "aligned-geodiff"),
        geodiff_bin=which("geodiff") if args.asp_dir is None else os.path.join(args.asp_dir, "geodiff"),
        verbose=args.verbose,
    )

    plot_gedi_coreg_results(
        gedi_gdf=gpd.read_file(args.gedi),
        dtm_path=args.dtm,
        before_geodiff_csv=original_geodiff_fn,
        after_geodiff_csv=aligned_geodiff_fn,
        shift_gdf=shift_gdf,
        outdir=args.outdir,
    )

if __name__ == "__main__":
    import subprocess
    main()