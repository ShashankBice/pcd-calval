#!/usr/bin/env python
"""
is2_dtm_coreg.py

Co-register ICESat-2 footprints to an airborne lidar DTM using ASP pc_align.
Aligns the entire dataset as a single point cloud.

I used ChatGSFC help to port my notebook code into this script, so if you see
any weirdness please let me know at shashank.bhushan@nasa.gov


Usage:
    python is2_dtm_coreg.py \
        --dtm /path/to/dtm.tif \
        --is2 /path/to/is2.gpkg \
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
import subprocess
from shutil import which

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


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

def apply_shift(gdf, dx, dy, dz, elevation_cols=None):
    """
    Apply a rigid translation to footprint positions and elevations.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input footprints with geometry and elevation column(s).
    dx, dy, dz : float
        Easting, northing, and vertical shifts in meters.
    elevation_cols : str or list of str, optional
        Name(s) of elevation column(s) to shift vertically.
        If None, no elevation columns are shifted.

    Returns
    -------
    GeoDataFrame
        Shifted footprints with updated geometry and elevation(s).
    """
    if elevation_cols is None:
        elevation_cols = []
    elif isinstance(elevation_cols, str):
        elevation_cols = [elevation_cols]

    crs = gdf.estimate_utm_crs()
    out = gdf.copy()
    for col in elevation_cols:
        if col in out.columns:
            out[col] = gdf[col] + dz
        else:
            print(f"  WARNING: elevation column '{col}' not found, skipping vertical shift for it")
    out["easting"] = gdf.geometry.x + dx
    out["northing"] = gdf.geometry.y + dy
    geometry = [Point(x, y) for x, y in zip(out["easting"], out["northing"])]
    out = gpd.GeoDataFrame(out.drop(columns=["geometry"]), geometry=geometry, crs=crs)
    return out


# =============================================================================
# Core co-registration function
# =============================================================================

def coreg_is2_to_dtm(
    dtm_path,
    is2_gpkg_path,
    outprefix,
    asp_bin_dir=None,
    alignment_method="point-to-plane",
    max_displacement=40,
    min_points=100,
    elevation_col="h_mean",
    verbose=False,
):
    """
    Co-register ICESat-2 footprints to a DTM using ASP pc_align.

    Aligns the entire dataset as a single point cloud rather than per-orbit.

    Parameters
    ----------
    dtm_path : str
        Path to reference DTM GeoTIFF.
    is2_gpkg_path : str
        Path to ICESat-2 footprints GeoPackage with columns:
        - median_ground (or as specified by elevation_col)
        - geometry (Point, in projected CRS matching DTM)
    outprefix : str
        Output prefix for alignment results.
    asp_bin_dir : str, optional
        Path to ASP bin directory. If None, assumes ASP tools are on PATH.
    alignment_method : str
        pc_align alignment method (default: 'point-to-plane').
    max_displacement : float
        Maximum expected displacement in meters.
    min_points : int
        Minimum number of footprints required to attempt alignment.
    elevation_col : str
        Column name for ICESat-2 terrain elevation.
    verbose : bool
        Print pc_align stdout/stderr.

    Returns
    -------
    aligned_gdf : GeoDataFrame
        Aligned ICESat-2 footprints.
    shift_record : dict
        Shift vector and metadata.
    """
    import rioxarray as rxr

    
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

    # Load DTM metadata for CRS/projection
    da_dtm = rxr.open_rasterio(dtm_path, masked=True).squeeze()
    dtm_crs = da_dtm.rio.crs

    # Write WKT for csv-srs
    wkt_path = f"{outprefix}-dtm_crs.wkt"
    with open(wkt_path, "w") as f:
        f.write(dtm_crs.to_wkt())

    csv_format = "1:easting,2:northing,3:height_above_datum"

    # Load ICESat-2 data
    gf_is2 = gpd.read_file(is2_gpkg_path)

    # Ensure projected CRS matches DTM
    if gf_is2.crs != dtm_crs:
        print(f"Reprojecting ICESat-2 from {gf_is2.crs} to {dtm_crs}")
        gf_is2 = gf_is2.to_crs(dtm_crs)

    gf_is2["easting"] = gf_is2.geometry.x
    gf_is2["northing"] = gf_is2.geometry.y

    num_points = len(gf_is2)
    centroid_x = gf_is2.geometry.x.mean()
    centroid_y = gf_is2.geometry.y.mean()

    print(f"Loaded {num_points} ICESat-2 footprints")

    # Build empty shift record template
    def _empty_record(status):
        return {
            "status": status,
            "num_points": num_points,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "dx_east_m": np.nan,
            "dy_north_m": np.nan,
            "dz_up_m": np.nan,
            "total_displacement_m": np.nan,
            "euler_north_deg": np.nan,
            "euler_east_deg": np.nan,
            "euler_down_deg": np.nan,
            "scale": np.nan,
        }

    # Check minimum point threshold
    if num_points < min_points:
        print(
            f"WARNING: Only {num_points} points, below minimum {min_points}. "
            f"Skipping alignment."
        )
        return gf_is2, _empty_record("skipped_insufficient_points")

    # Write CSV for pc_align
    gf_is2 = gf_is2.dropna(subset=[elevation_col])
    is2_csv = f"{outprefix}-is2_all_points.csv"
    gf_is2[["easting", "northing", elevation_col]].to_csv(
        is2_csv, index=False,
    )

    # Run pc_align
    align_prefix = f"{outprefix}-align_is2/is2_to_dtm"


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
        is2_csv,
        "-o", align_prefix,
    ]

    print(f"Aligning {num_points} ICESat-2 footprints to DTM...")
    result = _run_command(cmd, verbose=verbose)

    if result.returncode != 0:
        print(f"pc_align failed (returncode={result.returncode})")
        return gf_is2, _empty_record("pc_align_failed")

    # Parse alignment results
    log_files = sorted(glob.glob(f"{align_prefix}*log*pc_align*.txt"))
    if not log_files:
        print("No log file found")
        return gf_is2, _empty_record("no_log_found")

    params = parse_pc_align_log(log_files[-1])
    dx, dy, dz = params["translation_enu"]

    print(
        f"Alignment result: dx={dx:.2f} m, dy={dy:.2f} m, "
        f"dz={dz:.2f} m, total={params['total_displacement']:.2f} m"
    )

   # Apply shift
    aligned_gdf = apply_shift(gf_is2, dx, dy, dz, elevation_cols=elevation_col)
    shift_record = {
        "status": "success",
        "num_points": num_points,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "dx_east_m": dx,
        "dy_north_m": dy,
        "dz_up_m": dz,
        "total_displacement_m": params["total_displacement"],
        "euler_north_deg": params["euler_angles_ned"][0],
        "euler_east_deg": params["euler_angles_ned"][1],
        "euler_down_deg": params["euler_angles_ned"][2],
        "scale": params["scale"],
    }

    return aligned_gdf, shift_record


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
        description=(
            "Co-register ICESat-2 footprints to a DTM using ASP pc_align. "
            "Aligns the entire dataset as a single point cloud."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dtm", required=True,
        help="Path to reference DTM GeoTIFF.",
    )
    parser.add_argument(
        "--is2", required=True,
        help="Path to ICESat-2 ground-photon fit GeoPackage (projected CRS, "
             "with elevation column specified by --elevation-col). "
             "Used as source for co-registration against the DTM.",
    )
    parser.add_argument(
        "--is2-surface", default=None,
        help="Path to ICESat-2 canopy-photon fit GeoPackage. If provided, "
             "the shift derived from ground co-registration is applied to "
             "this file as well. Elevation column specified by "
             "--surface-elevation-col.",
    )
    parser.add_argument(
        "--surface-elevation-col", default="h_mean",
        help="Elevation column in the surface IS-2 file (default: h_mean).",
    )
    parser.add_argument(
        "--outprefix", required=True,
        help="Output prefix for alignment results (e.g. /path/to/output/is2_to_dtm). Will create directory if needed.",
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
        help="Minimum footprints required to attempt alignment (default: 100).",
    )
    parser.add_argument(
        "--elevation-col", default="h_mean",
        help="ICESat-2 elevation column name (default: h_mean).",
    )
    parser.add_argument(
        "--diff-clim", type=float, nargs=2, default=[-5, 5],
        metavar=("MIN", "MAX"),
        help="Symmetric color limits for difference plots (default: -5 5).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print pc_align stdout/stderr.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("ICESat-2 — DTM Co-registration")
    print(f"  DTM:           {args.dtm}")
    print(f"  IS-2 (ground): {args.is2}")
    if args.is2_surface:
        print(f"  IS-2 (surface): {args.is2_surface}")
    print(f"  Output Prefix:  {args.outprefix}")
    print("=" * 60)
    outdir = os.path.dirname(args.outprefix)
    if not os.path.exists(outdir):
        print(f"Creating output directory: {outdir}")
        os.makedirs(outdir)

    # ------------------------------------------------------------------
    # Run co-registration
    # ------------------------------------------------------------------
    aligned_gdf, shift_record = coreg_is2_to_dtm(
        dtm_path=args.dtm,
        is2_gpkg_path=args.is2,
        outprefix=args.outprefix,
        asp_bin_dir=args.asp_dir,
        alignment_method=args.alignment_method,
        max_displacement=args.max_displacement,
        min_points=args.min_points,
        elevation_col=args.elevation_col,
        verbose=args.verbose,
    )

    elevation_col = args.elevation_col

    # Save aligned footprints
    aligned_gpkg = f"{args.outprefix}-IS2_ground_aligned.gpkg"
    aligned_gdf.to_file(aligned_gpkg, driver="GPKG")
    print(f"Aligned footprints saved: {aligned_gpkg}")

    # Save shift record as GeoJSON
    shift_geojson = f"{args.outprefix}-coreg_shift_vector.geojson"
    shift_gdf = gpd.GeoDataFrame(
        [shift_record],
        geometry=[Point(shift_record["centroid_x"], shift_record["centroid_y"])],
        crs=aligned_gdf.crs,
    )
    shift_gdf.drop(columns=["centroid_x", "centroid_y"], inplace=True)
    shift_gdf.to_crs(epsg=4326).to_file(shift_geojson, driver="GeoJSON")
    print(f"Shift vector saved: {shift_geojson}")

    # ------------------------------------------------------------------
    # Apply same shift to surface IS-2 file if provided
    # ------------------------------------------------------------------
    aligned_surface_gdf = None
    if args.is2_surface is not None:
        import rioxarray as rxr

        crs = aligned_gdf.crs

        print(f"\nApplying ground-derived shift to surface IS-2 file...")
        print(f"  Surface file: {args.is2_surface}")

        gf_surface = gpd.read_file(args.is2_surface)
        if gf_surface.crs != crs:
            print(f"  Reprojecting surface IS-2 from {gf_surface.crs} to {crs}")
            gf_surface = gf_surface.to_crs(crs)

        print(f"  Loaded {len(gf_surface)} surface IS-2 segments")

        if shift_record["status"] == "success":
            dx = shift_record["dx_east_m"]
            dy = shift_record["dy_north_m"]
            dz = shift_record["dz_up_m"]

            # Shift all elevation-like columns in the surface file
            surface_elev_cols = [args.surface_elevation_col]
            # Also shift h_mean if it's different from the specified column
            for candidate in ["h_mean", "h_li"]:
                if candidate in gf_surface.columns and candidate not in surface_elev_cols:
                    surface_elev_cols.append(candidate)

            print(f"  Shifting elevation columns: {surface_elev_cols}")
            aligned_surface_gdf = apply_shift(
                gf_surface, dx, dy, dz, elevation_cols=surface_elev_cols
            )

            aligned_surface_gpkg = f"{args.outprefix}-IS2_surface_aligned.gpkg"
            aligned_surface_gdf.to_file(aligned_surface_gpkg, driver="GPKG")
            print(f"  Aligned surface footprints saved: {aligned_surface_gpkg}")

            print(
                f"  Applied shift: dx={dx:.2f} m, dy={dy:.2f} m, dz={dz:.2f} m "
                f"to {len(aligned_surface_gdf)} surface segments"
            )
        else:
            print(
                f"  WARNING: Ground alignment status='{shift_record['status']}'. "
                f"Saving surface file unshifted."
            )
            aligned_surface_gpkg = f"{args.outprefix}-IS2_surface_aligned.gpkg"
            gf_surface.to_file(aligned_surface_gpkg, driver="GPKG")
            aligned_surface_gdf = gf_surface

    # ------------------------------------------------------------------
    # Geodiff: before and after elevation differences
    # ------------------------------------------------------------------
    import rioxarray as rxr

    da_dtm = rxr.open_rasterio(args.dtm, masked=True).squeeze()
    dtm_crs = da_dtm.rio.crs
    wkt_path = f"{args.outprefix}-dtm_crs.wkt"
    csv_format = "1:easting,2:northing,3:height_above_datum"

    # Resolve geodiff binary
    def _find_tool(name):
        if args.asp_dir:
            path = os.path.join(args.asp_dir, name)
            if os.path.isfile(path):
                return path
        found = which(name)
        if found:
            return found
        raise FileNotFoundError(f"Cannot find '{name}'.")

    geodiff_bin = _find_tool("geodiff")

    # Load original IS-2 data in DTM CRS
    gf_is2_original = gpd.read_file(args.is2)
    if gf_is2_original.crs != dtm_crs:
        gf_is2_original = gf_is2_original.to_crs(dtm_crs)
    gf_is2_original["easting"] = gf_is2_original.geometry.x
    gf_is2_original["northing"] = gf_is2_original.geometry.y
    gf_is2_original = gf_is2_original.dropna(subset=[elevation_col])
    # Write original CSV
    original_csv =  f"{args.outprefix}-is2_original.csv"
    gf_is2_original[["easting", "northing", elevation_col]].to_csv(
        original_csv, index=False,
    )

    # Write aligned CSV
    aligned_csv = f"{args.outprefix}-is2_aligned.csv"
    aligned_gdf = aligned_gdf.dropna(subset=[elevation_col])
    aligned_gdf[["easting", "northing", elevation_col]].to_csv(
        aligned_csv, index=False,
    )

    # Geodiff: before alignment
    before_prefix = f"{args.outprefix}-geodiff_after"
    print(f"\nComputing geodiff (original ICESat-2 vs DTM)...")
    before_cmd = [
        geodiff_bin, original_csv, args.dtm,
        "--csv-format", csv_format,
        "--csv-srs", wkt_path,
        "-o", before_prefix,
    ]
    _run_command(before_cmd, verbose=args.verbose)
    before_geodiff_csv = glob.glob(f"{before_prefix}-diff.csv")
    before_geodiff_csv = before_geodiff_csv[0] if before_geodiff_csv else None

    # Geodiff: after alignment
    after_prefix = f"{args.outprefix}-geodiff_after"
    print(f"Computing geodiff (aligned ICESat-2 vs DTM)...")
    after_cmd = [
        geodiff_bin, aligned_csv, args.dtm,
        "--csv-format", csv_format,
        "--csv-srs", wkt_path,
        "-o", after_prefix,
    ]
    _run_command(after_cmd, verbose=args.verbose)
    after_geodiff_csv = glob.glob(f"{after_prefix}-diff.csv")
    after_geodiff_csv = after_geodiff_csv[0] if after_geodiff_csv else None

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Co-registration Summary")
    print("=" * 60)

    sr = shift_record
    print(f"  Status:       {sr['status']}")
    print(f"  N points:     {sr['num_points']:,}")

    if sr["status"] == "success":
        print(f"  dx (East):    {sr['dx_east_m']:.2f} m")
        print(f"  dy (North):   {sr['dy_north_m']:.2f} m")
        print(f"  dz (Up):      {sr['dz_up_m']:.2f} m")
        print(f"  Total shift:  {sr['total_displacement_m']:.2f} m")
        if aligned_surface_gdf is not None:
            print(f"  Surface file:  shift applied to {len(aligned_surface_gdf)} segments")

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    if before_geodiff_csv and after_geodiff_csv:
        from pcd_altimetry_coreg_plot import plot_is2_coreg_results

        print(f"\nGenerating co-registration summary figure...")
        plot_is2_coreg_results(
            is2_gdf=gf_is2_original,
            dtm_path=args.dtm,
            before_geodiff_csv=before_geodiff_csv,
            after_geodiff_csv=after_geodiff_csv,
            shift_record=shift_record,
            outprefix=args.outprefix,
            diff_clim=tuple(args.diff_clim),
            elevation_col=elevation_col,
            show=False,
        )
    else:
        print("\nWARNING: geodiff output not found, skipping plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()