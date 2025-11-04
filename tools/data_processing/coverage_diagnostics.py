#!/usr/bin/env python3
"""
Coverage Diagnostics for HARP-NOAA Mapping
===========================================

Comprehensive diagnostics following best practices:
- Label coverage by disk position (μ = cos(CMD))
- Match rates by flare class
- Time-to-nearest-frame distributions
- Limb bias detection

Based on analysis methods from Bobra & Couvidat and similar work.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def analyze_mapping_coverage(
    mapping_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Analyze HARP-NOAA mapping coverage.
    
    Returns dict with:
        - harp_coverage: % of HARPs with ≥1 NOAA mapping
        - noaa_per_harp_median: Median #NOAA ARs per HARP
        - match_type_breakdown: Counts by match type (id/spatial)
    """
    print("\n" + "="*70)
    print("HARP-NOAA Mapping Coverage Analysis")
    print("="*70)
    
    all_harps = set(frames_df['harpnum'].unique())
    mapped_harps = set(mapping_df['harpnum'].unique())
    
    harp_coverage = 100.0 * len(mapped_harps) / len(all_harps)
    
    noaa_per_harp = mapping_df.groupby('harpnum').size()
    noaa_per_harp_median = noaa_per_harp.median()
    
    print(f"\n[Mapping Statistics]")
    print(f"  Total HARPs in dataset: {len(all_harps)}")
    print(f"  HARPs with NOAA mapping: {len(mapped_harps)} ({harp_coverage:.1f}%)")
    print(f"  Total mappings: {len(mapping_df)}")
    print(f"  Unique NOAA ARs: {mapping_df['noaa_ar'].nunique()}")
    print(f"  Median NOAA ARs per HARP: {noaa_per_harp_median:.1f}")
    
    # Match type breakdown if available
    if 'match_type' in mapping_df.columns:
        match_types = mapping_df['match_type'].value_counts()
        print(f"\n[Match Type Breakdown]")
        for match_type, count in match_types.items():
            pct = 100.0 * count / len(mapping_df)
            print(f"  {match_type:10s}: {count:5d} ({pct:5.1f}%)")
    
    results = {
        'harp_coverage_pct': harp_coverage,
        'n_harps_total': len(all_harps),
        'n_harps_mapped': len(mapped_harps),
        'noaa_per_harp_median': noaa_per_harp_median,
        'total_mappings': len(mapping_df),
    }
    
    return results


def analyze_flare_coverage(
    flares_df: pd.DataFrame,
    mapping_df: pd.DataFrame
) -> dict:
    """
    Analyze what fraction of flares can be matched to HARPs.
    
    Breaks down by flare class (C/M/X).
    """
    print("\n" + "="*70)
    print("Flare Coverage Analysis")
    print("="*70)
    
    # Flares with NOAA AR
    has_noaa = flares_df['noaa_ar'].notna()
    n_has_noaa = has_noaa.sum()
    pct_has_noaa = 100.0 * n_has_noaa / len(flares_df)
    
    print(f"\n[Flare Statistics]")
    print(f"  Total flares: {len(flares_df)}")
    print(f"  Flares with NOAA AR: {n_has_noaa} ({pct_has_noaa:.1f}%)")
    
    # Flares that can be matched (NOAA AR in mapping)
    mapped_noaa_ars = set(mapping_df['noaa_ar'].unique())
    can_match = flares_df['noaa_ar'].isin(mapped_noaa_ars)
    n_matchable = can_match.sum()
    pct_matchable = 100.0 * n_matchable / len(flares_df)
    
    print(f"  Flares matchable to HARPs: {n_matchable} ({pct_matchable:.1f}%)")
    
    # Breakdown by class
    if 'class' in flares_df.columns:
        print(f"\n[Coverage by Flare Class]")
        for cls in ['C', 'M', 'X']:
            cls_flares = flares_df[flares_df['class'].str.startswith(cls, na=False)]
            if len(cls_flares) > 0:
                cls_matchable = cls_flares['noaa_ar'].isin(mapped_noaa_ars).sum()
                pct = 100.0 * cls_matchable / len(cls_flares)
                print(f"  {cls}-class: {cls_matchable}/{len(cls_flares)} ({pct:.1f}%)")
    
    results = {
        'n_flares_total': len(flares_df),
        'n_flares_with_noaa': n_has_noaa,
        'n_flares_matchable': n_matchable,
        'match_rate_pct': pct_matchable,
    }
    
    return results


def analyze_disk_position_bias(
    windows_df: pd.DataFrame,
    frames_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Analyze label coverage vs disk position (CMD or μ).
    
    Detects limb bias - are we missing labels at high CMD?
    """
    print("\n" + "="*70)
    print("Disk Position Bias Analysis")
    print("="*70)
    
    if 'has_noaa_mapping' not in windows_df.columns:
        print("  WARNING: 'has_noaa_mapping' column not found")
        return {}
    
    # Get CMD from window HARPs (requires joining with frames if not in windows)
    if 'cmd_deg' not in windows_df.columns and frames_df is not None:
        # Join windows with frames to get CMD
        # Use first frame for each window
        window_harps = windows_df[['harpnum', 't0', 'has_noaa_mapping']].copy()
        frames_near_t0 = []
        
        for _, win in window_harps.iterrows():
            frame = frames_df[
                (frames_df['harpnum'] == win['harpnum']) &
                (frames_df['date_obs'] <= win['t0'])
            ].sort_values('date_obs').iloc[-1] if len(frames_df[
                (frames_df['harpnum'] == win['harpnum']) &
                (frames_df['date_obs'] <= win['t0'])
            ]) > 0 else None
            
            if frame is not None:
                frames_near_t0.append({
                    'harpnum': win['harpnum'],
                    't0': win['t0'],
                    'cmd_deg': frame['cmd_deg'],
                    'has_noaa_mapping': win['has_noaa_mapping']
                })
        
        analysis_df = pd.DataFrame(frames_near_t0)
    elif 'cmd_deg' in windows_df.columns:
        analysis_df = windows_df[['cmd_deg', 'has_noaa_mapping']].copy()
    else:
        print("  WARNING: Cannot compute CMD - need 'cmd_deg' column or frames_df")
        return {}
    
    if len(analysis_df) == 0:
        print("  No data for analysis")
        return {}
    
    # Bin by CMD
    analysis_df['cmd_bin'] = pd.cut(
        analysis_df['cmd_deg'].abs(),
        bins=[0, 30, 45, 60, 70, 90],
        labels=['0-30°', '30-45°', '45-60°', '60-70°', '70-90°']
    )
    
    print(f"\n[Label Coverage by Disk Position (|CMD|)]")
    print(f"{'CMD Range':12s}  {'Total':>8s}  {'Labeled':>8s}  {'Coverage':>10s}")
    print("-" * 50)
    
    for bin_label in ['0-30°', '30-45°', '45-60°', '60-70°', '70-90°']:
        bin_data = analysis_df[analysis_df['cmd_bin'] == bin_label]
        if len(bin_data) > 0:
            n_labeled = bin_data['has_noaa_mapping'].sum()
            pct = 100.0 * n_labeled / len(bin_data)
            print(f"{bin_label:12s}  {len(bin_data):8d}  {n_labeled:8d}  {pct:9.1f}%")
    
    # Plot if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        coverage_by_bin = analysis_df.groupby('cmd_bin')['has_noaa_mapping'].agg(['sum', 'count'])
        coverage_by_bin['pct'] = 100.0 * coverage_by_bin['sum'] / coverage_by_bin['count']
        
        ax.bar(range(len(coverage_by_bin)), coverage_by_bin['pct'])
        ax.set_xticks(range(len(coverage_by_bin)))
        ax.set_xticklabels(coverage_by_bin.index, rotation=45)
        ax.set_ylabel('Label Coverage (%)')
        ax.set_xlabel('|CMD| Range')
        ax.set_title('Label Coverage vs Disk Position')
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'coverage_vs_cmd.png', dpi=150)
        print(f"\n  Saved plot: {output_dir / 'coverage_vs_cmd.png'}")
        plt.close()
    
    return {
        'coverage_by_cmd': coverage_by_bin['pct'].to_dict() if 'coverage_by_bin' in locals() else {}
    }


def generate_full_diagnostic_report(
    mapping_df: pd.DataFrame,
    flares_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    output_dir: Path
):
    """Generate comprehensive diagnostic report."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Generating Full Diagnostic Report")
    print("="*70)
    
    # Run all analyses
    mapping_stats = analyze_mapping_coverage(mapping_df, frames_df, output_dir)
    flare_stats = analyze_flare_coverage(flares_df, mapping_df)
    disk_stats = analyze_disk_position_bias(windows_df, frames_df, output_dir)
    
    # Combine into report
    report = {
        'mapping': mapping_stats,
        'flares': flare_stats,
        'disk_position': disk_stats,
    }
    
    # Save JSON report
    import json
    report_path = output_dir / 'coverage_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Saved diagnostic report: {report_path}")
    
    return report

