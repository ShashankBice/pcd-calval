"""
pcd_altimetry-coreg-plot.py

Plotting utilities for PCD elevation data co-registration results.
Designed to be called from gedi_dtm_coreg.py or used standalone.

Usage from gedi_dtm_coreg.py:
    from pcd_altimetry-coreg-plot import plot_gedi_coreg_results
    plot_gedi_coreg_results(
        gedi_gdf=gf_gedi_filtered_proj,
        dtm_path="workunit-DTM.tif",
        before_geodiff_csv="non_aligned-diff.csv",
        after_geodiff_csv="aligned-diff.csv",
        shift_gdf=shift_gdf,
        outdir="./figures",
    )
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
from osgeo import gdal
import geopandas as gpd
import contextily as ctx


# =============================================================================
# Utilities
# =============================================================================

def nmad(values):
    """Normalized Median Absolute Deviation (robust spread estimator)."""
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def get_clim(ar):
    """Return 2nd/98th percentile clip for display."""
    try:
        clim = np.percentile(ar.compressed(), (2, 98))
    except AttributeError:
        clim = np.percentile(ar, (2, 98))
    return tuple(clim)


def symmetric_clim(ar1, ar2):
    """Symmetric color limits spanning two arrays."""
    perc1 = get_clim(ar1)
    perc2 = get_clim(ar2)
    abs_max = max(abs(perc1[0]), abs(perc1[1]), abs(perc2[0]), abs(perc2[1]))
    return (-abs_max, abs_max)


def fn_to_ma(fn, band=1):
    """Read a GeoTIFF band as a masked array."""
    with rasterio.open(fn) as ds:
        ar = ds.read(band)
        ndv = ds.nodatavals[0]
        if ndv is None:
            ndv = ar[0, 0]
    return np.ma.masked_equal(ar, ndv)


def make_hillshade(dem_path):
    """Generate an in-memory hillshade array from a DEM path."""
    ds = gdal.Open(dem_path)
    hs_ds = gdal.DEMProcessing("", ds, "hillshade", format="MEM")
    hs = hs_ds.ReadAsArray()
    ds = None
    hs_ds = None
    return hs


def read_geodiff_csv(csv_fn):
    """
    Read ASP geodiff output CSV as a GeoDataFrame.

    Parameters
    ----------
    csv_fn : str
        Path to geodiff -diff.csv output.

    Returns
    -------
    GeoDataFrame with columns: lon, lat, diff, geometry (EPSG:4326).
    """
    cols = ["lon", "lat", "diff"]
    df = pd.read_csv(csv_fn, comment="#", names=cols)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"),
    )
    return gdf


def plot_ar(im, ax, clim, cmap=None, label=None, cbar=True, alpha=1):
    """Plot a 2D array with optional colorbar."""
    kwargs = dict(clim=clim, alpha=alpha, interpolation="none")
    if cmap:
        kwargs["cmap"] = cmap
    img = ax.imshow(im, **kwargs)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad="1.5%")
        cb = plt.colorbar(img, cax=cax, ax=ax, extend="both")
        if label:
            cax.set_ylabel(label)
    ax.set_xticks([])
    ax.set_yticks([])
    return img


# =============================================================================
# Main GEDI co-registration plot
# =============================================================================

def plot_gedi_coreg_results(
    gedi_gdf,
    dtm_path,
    before_geodiff_csv,
    after_geodiff_csv,
    shift_gdf=None,
    outdir=None,
    plot_crs=None,
    diff_clim=(-5, 5),
    basemap_provider=None,
    markersize=2,
    elevation_col="elevation_lm",
    figname_prefix="gedi_coreg",
    dpi=200,
    show=True,
):
    """
    Plot GEDI-DTM co-registration results as a single figure:
    top row (2x2 map panels), bottom row (histogram spanning full width).

    Parameters
    ----------
    gedi_gdf : GeoDataFrame
        Original (pre-alignment) GEDI footprints with elevation_col column.
    dtm_path : str
        Path to reference DTM GeoTIFF.
    before_geodiff_csv : str
        Path to geodiff CSV (GEDI original - DTM).
    after_geodiff_csv : str
        Path to geodiff CSV (GEDI aligned - DTM).
    shift_gdf : GeoDataFrame, optional
        Shift vectors from coreg_gedi_to_dtm (one row per orbit).
        If provided, per-orbit shifts are displayed in the after-alignment title.
    outdir : str, optional
        Directory to save figure. If None, figure is not saved.
    plot_crs : CRS, optional
        CRS for map display. Defaults to DTM CRS.
    diff_clim : tuple
        Symmetric color limits for difference maps (default: (-5, 5)).
    basemap_provider : contextily provider, optional
        Basemap tile provider. Default: Esri.WorldImagery.
    markersize : float
        Marker size for point plots.
    elevation_col : str
        GEDI elevation column name.
    figname_prefix : str
        Prefix for saved figure filename.
    dpi : int
        Figure resolution for saving.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : Figure
        The combined figure.
    """
    if basemap_provider is None:
        basemap_provider = ctx.providers.Esri.WorldImagery

    # Read geodiff residuals
    before_diff = read_geodiff_csv(before_geodiff_csv)
    after_diff = read_geodiff_csv(after_geodiff_csv)

    # Determine plot CRS
    if plot_crs is None:
        with rasterio.open(dtm_path) as ds:
            plot_crs = ds.crs

    # Reproject to plot CRS
    before_diff = before_diff.to_crs(plot_crs)
    after_diff = after_diff.to_crs(plot_crs)
    gedi_plot = gedi_gdf.to_crs(plot_crs)

    # Read DTM for hillshade + elevation overlay
    dtm_ma = fn_to_ma(dtm_path)
    hs = make_hillshade(dtm_path)

    # Compute per-orbit shift summary for title
    shift_lines = []
    if shift_gdf is not None:
        success = shift_gdf[shift_gdf["status"] == "success"]
        if len(success) > 0:
            for _, row in success.iterrows():
                shift_lines.append(
                    f"orbit {int(row['orbit'])}: "
                    f"dx={row['dx_east_m']:.2f}, "
                    f"dy={row['dy_north_m']:.2f}, "
                    f"dz={row['dz_up_m']:.2f} m"
                )

    # Compute histogram stats
    before_vals = before_diff["diff"].dropna().values
    after_vals = after_diff["diff"].dropna().values
    before_med = np.median(before_vals)
    before_nmad = nmad(before_vals)
    after_med = np.median(after_vals)
    after_nmad = nmad(after_vals)

    # ---- Combined figure: 3 rows x 2 cols ----
    # Top two rows: 2x2 map panels
    # Bottom row: histogram spanning both columns
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.25, wspace=0.2)

    ax_gedi = fig.add_subplot(gs[0, 0])
    ax_dtm = fig.add_subplot(gs[0, 1])
    ax_before = fig.add_subplot(gs[1, 0])
    ax_after = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[2, :])

    # --- Panel 1: GEDI footprints on basemap ---
    if len(gedi_plot) > 10000:
        gedi_sample = gedi_plot.sample(10000, random_state=42)
    else:
        gedi_sample = gedi_plot
    gedi_sample.plot(
        column=elevation_col, ax=ax_gedi, markersize=markersize,
        legend=True,
        legend_kwds={"label": f"{elevation_col} (m)", "shrink": 0.6},
    )
    try:
        ctx.add_basemap(
            ax=ax_gedi, crs=plot_crs, source=basemap_provider,
            attribution=False,
        )
    except Exception as e:
        print(f"  Basemap failed for GEDI panel: {e}")
    ax_gedi.set_title(f"GEDI footprints (n={len(gedi_plot):,})")
    ax_gedi.set_xticks([])
    ax_gedi.set_yticks([])

    # --- Panel 2: DTM hillshade + elevation ---
    ax_dtm.imshow(hs, cmap="gray", clim=get_clim(hs), interpolation="none")
    plot_ar(
        dtm_ma, ax=ax_dtm, clim=get_clim(dtm_ma),
        label="Elevation (m)", alpha=0.6,
    )
    ax_dtm.set_title("Reference ALS DTM")

    # --- Panel 3: Before alignment ---
    before_diff.plot(
        column="diff", ax=ax_before, cmap="RdBu",
        vmin=diff_clim[0], vmax=diff_clim[1], markersize=markersize,
    )
    try:
        ctx.add_basemap(
            ax=ax_before, crs=plot_crs, source=basemap_provider,
            attribution=False,
        )
    except Exception:
        pass
    ax_before.set_title("Before alignment")
    ax_before.set_xticks([])
    ax_before.set_yticks([])

    # --- Panel 4: After alignment with shift vectors in title ---
    after_diff.plot(
        column="diff", ax=ax_after, cmap="RdBu",
        vmin=diff_clim[0], vmax=diff_clim[1], markersize=markersize,
    )
    try:
        ctx.add_basemap(
            ax=ax_after, crs=plot_crs, source=basemap_provider,
            attribution=False,
        )
    except Exception:
        pass

    if shift_lines:
        after_title = "After alignment\n" + "\n".join(shift_lines)
    else:
        after_title = "After alignment"
    ax_after.set_title(after_title, fontsize=7, loc="left", family="monospace")
    ax_after.set_xticks([])
    ax_after.set_yticks([])

    # --- Panel 5: Histogram (full width) ---
    bins = np.linspace(diff_clim[0], diff_clim[1], 128)

    ax_hist.hist(
        before_vals, bins=bins, color="steelblue", alpha=0.6,
        label=(
            f"Before (n={len(before_vals):,}): "
            f"med={before_med:.2f} m, NMAD={before_nmad:.2f} m"
        ),
    )
    ax_hist.hist(
        after_vals, bins=bins, color="seagreen", alpha=0.6,
        label=(
            f"After (n={len(after_vals):,}): "
            f"med={after_med:.2f} m, NMAD={after_nmad:.2f} m"
        ),
    )
    ax_hist.axvline(x=0, linestyle="--", linewidth=1, color="k", alpha=0.7)
    ax_hist.set_xlabel("Elevation difference (m)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=8)
    ax_hist.set_title("GEDI − DTM elevation residuals")
    plt.tight_layout()
    # ---- Save ----
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig_fn = os.path.join(outdir, f"{figname_prefix}_summary.png")
        fig.savefig(fig_fn, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {fig_fn}")

    if show:
        plt.show()

    return fig


# =============================================================================
# Standalone raster alignment plot (for DEM-to-DEM comparisons)
# =============================================================================

def plot_alignment_maps(
    ref_dem_path,
    src_dem_path,
    before_diff_fn,
    after_diff_fn,
    outdir=None,
    figname_prefix="dem_alignment",
    dpi=200,
    show=True,
):
    """
    Plot raster DEM-to-DEM alignment results (before/after difference maps).

    Parameters
    ----------
    ref_dem_path, src_dem_path : str
        Paths to reference and source DEMs.
    before_diff_fn, after_diff_fn : str
        Paths to elevation difference rasters (before and after alignment).
    outdir : str, optional
        Save directory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    ax_ref, ax_src, ax_before, ax_after = axes.ravel()

    ref_ma = fn_to_ma(ref_dem_path)
    src_ma = fn_to_ma(src_dem_path)
    before_ma = fn_to_ma(before_diff_fn)
    after_ma = fn_to_ma(after_diff_fn)

    ref_hs = make_hillshade(ref_dem_path)
    src_hs = make_hillshade(src_dem_path)

    # Reference DEM
    ax_ref.imshow(
        ref_hs, cmap="gray", clim=get_clim(ref_hs), interpolation="none",
    )
    plot_ar(
        ref_ma, ax=ax_ref, clim=get_clim(ref_ma),
        label="Elevation (m)", alpha=0.6,
    )
    ax_ref.set_title("Reference DEM")

    # Source DEM
    ax_src.imshow(
        src_hs, cmap="gray", clim=get_clim(src_hs), interpolation="none",
    )
    plot_ar(
        src_ma, ax=ax_src, clim=get_clim(src_ma),
        label="Elevation (m)", alpha=0.6,
    )
    ax_src.set_title("Source DEM")

    # Differences
    diff_clim = symmetric_clim(before_ma, after_ma)
    plot_ar(
        before_ma, ax=ax_before, clim=diff_clim, cmap="RdBu", label="Δh (m)",
    )
    ax_before.set_title("Before alignment")

    plot_ar(
        after_ma, ax=ax_after, clim=diff_clim, cmap="RdBu", label="Δh (m)",
    )
    ax_after.set_title("After alignment")

    fig.tight_layout()

    # Histogram
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
    bins = np.linspace(diff_clim[0], diff_clim[1], 128)

    before_med = np.median(before_ma.compressed())
    before_nmad_val = nmad(before_ma.compressed())
    after_med = np.median(after_ma.compressed())
    after_nmad_val = nmad(after_ma.compressed())

    ax_hist.hist(
        before_ma.compressed(), bins=bins, color="steelblue", alpha=0.6,
        label=f"Before: med={before_med:.2f} m, NMAD={before_nmad_val:.2f} m",
    )
    ax_hist.hist(
        after_ma.compressed(), bins=bins, color="seagreen", alpha=0.6,
        label=f"After: med={after_med:.2f} m, NMAD={after_nmad_val:.2f} m",
    )
    ax_hist.axvline(x=0, linestyle="--", linewidth=1, color="k", alpha=0.7)
    ax_hist.set_xlabel("Elevation difference (m)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=8)
    ax_hist.set_title("Elevation residuals")
    fig_hist.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(
            os.path.join(outdir, f"{figname_prefix}_maps.png"),
            dpi=dpi, bbox_inches="tight",
        )
        fig_hist.savefig(
            os.path.join(outdir, f"{figname_prefix}_histogram.png"),
            dpi=dpi, bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, fig_hist



def plot_is2_coreg_results(
    is2_gdf,
    dtm_path,
    before_geodiff_csv,
    after_geodiff_csv,
    shift_record=None,
    outprefix=None,
    plot_crs=None,
    diff_clim=(-5, 5),
    basemap_provider=None,
    markersize=0.5,
    elevation_col="median_ground",
    figname_prefix="is2_coreg",
    dpi=300,
    show=True,
):
    """
    Plot ICESat-2 — DTM co-registration results as a single figure:
    top row (2x2 map panels), bottom row (histogram spanning full width).

    Parameters
    ----------
    is2_gdf : GeoDataFrame
        Original (pre-alignment) ICESat-2 footprints.
    dtm_path : str
        Path to reference DTM GeoTIFF.
    before_geodiff_csv : str
        Path to geodiff CSV (IS-2 original - DTM).
    after_geodiff_csv : str
        Path to geodiff CSV (IS-2 aligned - DTM).
    shift_record : dict, optional
        Shift vector from coreg_is2_to_dtm.
        If provided, shift is displayed in the after-alignment title.
    outprefix : str, optional
        Outprefix to save figure. If None, figure is not saved.
    plot_crs : CRS, optional
        CRS for map display. Defaults to DTM CRS.
    diff_clim : tuple
        Symmetric color limits for difference maps (default: (-5, 5)).
    basemap_provider : contextily provider, optional
        Basemap tile provider. Default: Esri.WorldImagery.
    markersize : float
        Marker size for point plots (default: 0.5, smaller for dense IS-2).
    elevation_col : str
        ICESat-2 elevation column name.
    figname_prefix : str
        Prefix for saved figure filename.
    dpi : int
        Figure resolution for saving.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : Figure
        The combined figure.
    """
    if basemap_provider is None:
        basemap_provider = ctx.providers.Esri.WorldImagery

    # Read geodiff residuals
    before_diff = read_geodiff_csv(before_geodiff_csv)
    after_diff = read_geodiff_csv(after_geodiff_csv)

    # Determine plot CRS
    if plot_crs is None:
        with rasterio.open(dtm_path) as ds:
            plot_crs = ds.crs

    # Reproject to plot CRS
    before_diff = before_diff.to_crs(plot_crs)
    after_diff = after_diff.to_crs(plot_crs)
    is2_plot = is2_gdf.to_crs(plot_crs)

    # Read DTM for hillshade + elevation overlay
    dtm_ma = fn_to_ma(dtm_path)
    hs = make_hillshade(dtm_path)

    # Build shift title
    shift_title = "After alignment"
    if shift_record is not None and shift_record.get("status") == "success":
        shift_title += (
            f"\ndx={shift_record['dx_east_m']:.2f} m, "
            f"dy={shift_record['dy_north_m']:.2f} m, "
            f"dz={shift_record['dz_up_m']:.2f} m "
            f"(total={shift_record['total_displacement_m']:.2f} m)"
        )

    # Compute histogram stats
    before_vals = before_diff["diff"].dropna().values
    after_vals = after_diff["diff"].dropna().values
    before_med = np.median(before_vals)
    before_nmad = nmad(before_vals)
    after_med = np.median(after_vals)
    after_nmad = nmad(after_vals)

    # ---- Combined figure: 3 rows x 2 cols ----
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1, 1, 0.6], hspace=0.25, wspace=0.2,
    )

    ax_is2 = fig.add_subplot(gs[0, 0])
    ax_dtm = fig.add_subplot(gs[0, 1])
    ax_before = fig.add_subplot(gs[1, 0])
    ax_after = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[2, :])

    # --- Panel 1: ICESat-2 footprints on basemap ---
    if len(is2_plot) > 20000:
        is2_sample = is2_plot.sample(20000, random_state=42)
    else:
        is2_sample = is2_plot
    is2_sample.plot(
        column=elevation_col, ax=ax_is2, markersize=markersize,
        legend=True,
        legend_kwds={"label": f"{elevation_col} (m)", "shrink": 0.6},
    )
    try:
        ctx.add_basemap(
            ax=ax_is2, crs=plot_crs, source=basemap_provider,
            attribution=False,
        )
    except Exception as e:
        print(f"  Basemap failed for IS-2 panel: {e}")
    ax_is2.set_title(f"ICESat-2 footprints (n={len(is2_plot):,})")
    ax_is2.set_xticks([])
    ax_is2.set_yticks([])

    # --- Panel 2: DTM hillshade + elevation ---
    ax_dtm.imshow(hs, cmap="gray", clim=get_clim(hs), interpolation="none")
    plot_ar(
        dtm_ma, ax=ax_dtm, clim=get_clim(dtm_ma),
        label="Elevation (m)", alpha=0.6,
    )
    ax_dtm.set_title("Reference ALS DTM")

    # --- Panel 3: Before alignment ---
    before_diff.plot(
        column="diff", ax=ax_before, cmap="RdBu",
        vmin=diff_clim[0], vmax=diff_clim[1], markersize=markersize,
    )
    try:
        ctx.add_basemap(
            ax=ax_before, crs=plot_crs, source=basemap_provider,
            attribution=False,
        )
    except Exception:
        pass
    ax_before.set_title("Before alignment")
    ax_before.set_xticks([])
    ax_before.set_yticks([])

    # --- Panel 4: After alignment with shift in title ---
    after_diff.plot(
        column="diff", ax=ax_after, cmap="RdBu",
        vmin=diff_clim[0], vmax=diff_clim[1], markersize=markersize,
    )
    try:
        ctx.add_basemap(
            ax=ax_after, crs=plot_crs, source=basemap_provider,
            attribution=False,
        )
    except Exception:
        pass
    ax_after.set_title(shift_title, fontsize=8, loc="left", family="monospace")
    ax_after.set_xticks([])
    ax_after.set_yticks([])

    # --- Panel 5: Histogram (full width) ---
    bins = np.linspace(diff_clim[0], diff_clim[1], 128)

    ax_hist.hist(
        before_vals, bins=bins, color="steelblue", alpha=0.6,
        label=(
            f"Before (n={len(before_vals):,}): "
            f"med={before_med:.2f} m, NMAD={before_nmad:.2f} m"
        ),
    )
    ax_hist.hist(
        after_vals, bins=bins, color="seagreen", alpha=0.6,
        label=(
            f"After (n={len(after_vals):,}): "
            f"med={after_med:.2f} m, NMAD={after_nmad:.2f} m"
        ),
    )
    ax_hist.axvline(x=0, linestyle="--", linewidth=1, color="k", alpha=0.7)
    ax_hist.set_xlabel("Elevation difference (m)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=8)
    ax_hist.set_title("ICESat-2 − DTM elevation residuals")

    # ---- Save ----
    if outprefix:
        #os.makedirs(outdir, exist_ok=True)
        fig_fn = f"{outprefix}-{figname_prefix}_summary.png"
        fig.savefig(fig_fn, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {fig_fn}")

    if show:
        plt.show()

    return fig