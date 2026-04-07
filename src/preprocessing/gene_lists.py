# ---------------------------------------------------------------------------
# Cell-type marker gene sets for cluster annotation
# ---------------------------------------------------------------------------

# Elmentaite et al. 2021 + literature — gut cell types
GUT_CELL_TYPE_MARKERS: dict[str, list[str]] = {
    # Epithelial
    "Stem/TA":          ["LGR5", "OLFM4", "ASCL2", "AXIN2", "MKI67", "TOP2A", "PCNA", "SOX9"],
    "Enterocyte":       ["FABP1", "FABP2", "ALPI", "SLC5A1", "APOA4", "APOB", "CYP3A4", "ALDOB"],
    "Goblet":           ["MUC2", "TFF3", "CLCA1", "SPDEF", "AGR2", "FCGBP"],
    "Enteroendocrine":  ["CHGA", "CHGB", "NEUROD1", "PAX4", "NEUROG3", "SST", "GCG"],
    "Paneth":           ["DEFA5", "DEFA6", "LYZ", "MMP7", "ITLN2"],
    "Tuft":             ["DCLK1", "POU2F3", "TRPM5", "PTGS1", "IL25"],
    "M cell":           ["GP2", "SPIB", "CCL20", "MARCKS"],
    # Stromal
    "Fibroblast":       ["COL1A1", "COL3A1", "DCN", "LUM", "PDGFRA", "VIM"],
    "Endothelial":      ["PECAM1", "CDH5", "VWF", "CLDN5", "ENG", "PLVAP"],
    "Smooth muscle":    ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYLK"],
    "Pericyte":         ["RGS5", "PDGFRB", "NOTCH3", "MCAM", "ABCC9"],
    # Immune
    "Macrophage":       ["CD68", "LYZ", "CSF1R", "MRC1", "MARCO", "MSR1"],
    "Dendritic cell":   ["CLEC9A", "XCR1", "ITGAX", "FCER1A", "CLEC10A"],
    "T cell":           ["CD3D", "CD3E", "TRAC", "CD4", "CD8A", "IL7R"],
    "B cell":           ["MS4A1", "CD79A", "CD79B", "IGHM", "BANK1"],
    "Plasma cell":      ["MZB1", "SDC1", "IGHG1", "IGHA1", "JCHAIN", "DERL3"],
    "NK/ILC":           ["GNLY", "NKG7", "KLRD1", "RORC", "GATA3", "EOMES"],
    "Mast cell":        ["TPSAB1", "CPA3", "KIT", "HPGDS", "MS4A2"],
}

# 10x Genomics PBMC3k + literature — peripheral blood cell types
PBMC_CELL_TYPE_MARKERS: dict[str, list[str]] = {
    # T cells
    "CD4+ T cell":        ["CD3D", "CD3E", "CD4", "IL7R", "TCF7", "LEF1"],
    "CD8+ T cell":        ["CD3D", "CD3E", "CD8A", "CD8B", "GZMK", "CCL5"],
    "Naive T cell":       ["CCR7", "SELL", "LEF1", "TCF7", "IL7R", "S100A12"],
    "Memory T cell":      ["IL7R", "S100A4", "AQP3", "GPR183"],
    "Regulatory T":       ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2"],
    # NK cells
    "NK cell":            ["GNLY", "NKG7", "KLRD1", "GZMB", "NCAM1", "FCGR3A"],
    # B cells
    "B cell":             ["MS4A1", "CD79A", "CD79B", "IGHM", "IGHD", "BANK1"],
    "Plasma cell":        ["MZB1", "JCHAIN", "IGHG1", "IGHA1", "SDC1", "DERL3"],
    # Monocytes
    "Classical Mono":     ["CD14", "LYZ", "S100A8", "S100A9", "CSF1R", "VCAN"],
    "Non-classical Mono": ["FCGR3A", "MS4A7", "VMO1", "LST1", "AIF1"],
    # Dendritic cells
    "Plasmacytoid DC":    ["IL3RA", "GZMB", "ITM2C", "LILRA4", "CLEC4C"],
    "Myeloid DC":         ["FCER1A", "CST3", "CLEC9A", "XCR1", "CLEC10A"],
    # Other
    "Megakaryocyte":      ["PPBP", "PF4", "GP1BA", "ITGA2B", "TUBB1"],
}


# ---------------------------------------------------------------------------
# Y-chromosome and X-inactivation genes commonly used for sex regression
SEX_GENES = [
    'XIST', 'RPS4Y1', 'DDX3Y', 'USP9Y', 'KDM5D', 'EIF1AY', 'ZFY', 'PRKY', 'TXLNGY',
]

# Tirosh et al. 2016 (Science) — S-phase and G2/M-phase gene lists
S_GENES = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2',
    'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2',
    'NASP', 'RAD51AP1', 'GMNN', 'WDC45', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2',
    'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM',
    'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8',
]

G2M_GENES = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2',
    'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2',
    'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1',
    'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1',
    'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'SSNA1',
    'FOXM1', 'CBFB', 'CENPA', 'CKAP5', 'UHRF2', 'KIF20A', 'CENPE', 'CTCF', 'NEK2',
    'G2E3', 'GAS2L3', 'CBX5',
]
