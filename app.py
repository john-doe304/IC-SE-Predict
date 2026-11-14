# -----------------------------------------------------------
# Solid Electrolyte Ionic Conductivity Predictor
# Optimized Streamlit application
# -----------------------------------------------------------

import os
import re
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Optional structure-visualization dependencies.
try:
    import py3Dmol
except Exception:
    py3Dmol = None

try:
    from mp_api.client import MPRester
except Exception:
    MPRester = None

try:
    from pymatgen.core import Composition, Structure, Lattice
    from pymatgen.io.cif import CifWriter
except Exception:
    Composition = Structure = Lattice = CifWriter = None

try:
    from autogluon.tabular import TabularPredictor
except Exception:
    TabularPredictor = None

try:
    from matminer.featurizers.composition import ElementProperty, Stoichiometry
except Exception:
    ElementProperty = Stoichiometry = None


st.set_page_config(
    page_title="Ionic Conductivity Predictor",
    page_icon="🔋",
    layout="wide",
)

MODEL_PATH = "./ag-20251024_075719"
MODEL_NAMES = [
    "CatBoost",
    "ExtraTreesMSE",
    "LightGBM",
    "KNeighborsDist",
    "WeightedEnsemble_L2",
    "XGBoost",
]

# Keep the feature names identical to those used when training the model.
REQUIRED_DESCRIPTORS = [
    "MagpieData mean CovalentRadius",
    "Temp",
    "MagpieData avg_dev SpaceGroupNumber",
    "0-norm",
    "MagpieData mean MeltingT",
    "MagpieData avg_dev Column",
    "MagpieData mean NValence",
]

MP_COLORS = {
    "H": "#FFFFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#909090",
    "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F",
    "K": "#8F40D4", "Ca": "#FFD478", "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB",
    "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050",
    "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#4C4CFF", "As": "#BD80E3",
    "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1", "Rb": "#702EB0", "Sr": "#00FF00",
    "Y": "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9", "Mo": "#54B5B5", "Ru": "#248F8F",
    "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573",
    "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Xe": "#4DC4FF",
    "Cs": "#57178F", "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7",
    "Nd": "#C7FFC7", "Pm": "#A3FFC7", "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7",
    "Tb": "#30FFC7", "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452",
    "Yb": "#00BF69", "Lu": "#00AB6B", "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6",
    "Re": "#267DAB", "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123",
    "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00",
    "At": "#754F45", "Rn": "#428296", "Fr": "#420066", "Ra": "#00C900", "Ac": "#70ABFA",
    "Th": "#00BAFF", "Pa": "#00A1FF", "U": "#008FFF", "Np": "#0080FF", "Pu": "#006BFF",
}


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
