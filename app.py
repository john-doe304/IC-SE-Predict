# -----------------------------------------------------------
#   Solid Electrolyte Ionic Conductivity Predictor (Final Fixed)
#   With Materials Project Crystal Rendering + Fallback Cell
# -----------------------------------------------------------

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import py3Dmol
import traceback
import gc
import re
import tempfile
import os
import random
from io import BytesIO
import base64

# rdkit, mordred, autogluon, matminer may be heavy — import guarded
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, AllChem
    from rdkit.Chem.Draw import MolDraw2DSVG
    from rdkit.ML.Descriptors import MoleculeDescriptors
except Exception:
    rdkit = None

try:
    from mordred import Calculator, descriptors
except Exception:
    pass

try:
    from autogluon.tabular import TabularPredictor
except Exception:
    TabularPredictor = None

# matminer descriptors (guarded import)
try:
    from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry, IonProperty
    from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
except Exception:
    ElementProperty = Meredig = Stoichiometry = IonProperty = StrToComposition = CompositionToOxidComposition = None

# Materials Project (mp-api) and pymatgen
try:
    from mp_api.client import MPRester
except Exception:
    MPRester = None

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.cif import CifWriter
except Exception:
    Structure = Lattice = CifWriter = None

st.set_page_config(layout="wide", page_title="Ionic Conductivity Predictor")

# ----------------------------------
# MP 官方配色（可扩充）
# ----------------------------------
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
    "Th": "#00BAFF", "Pa": "#00A1FF", "U": "#008FFF", "Np": "#0080FF", "Pu": "#006BFF"
}

# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 40%;
        background-color: #f9f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }
