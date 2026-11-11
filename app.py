# -----------------------------------------------------------
#   Solid Electrolyte Ionic Conductivity Predictor (Revised with Uncertainty)
#   With Materials Project Crystal Rendering + Fallback Cell
# -----------------------------------------------------------

import base64
import gc
import os
import random
import re
import tempfile
import traceback
from io import BytesIO
import numpy as np
import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components

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
  from matminer.featurizers.composition import (
      ElementProperty,
      IonProperty,
      Meredig,
      Stoichiometry,
  )
  from matminer.featurizers.conversions import (
      CompositionToOxidComposition,
      StrToComposition,
  )
except Exception:
  ElementProperty = (
      Meredig
  ) = (
      Stoichiometry
  ) = IonProperty = StrToComposition = CompositionToOxidComposition = None

# Materials Project (mp-api) and pymatgen
try:
  from mp_api.client import MPRester
except Exception:
  MPRester = None

try:
  from pymatgen.core import Lattice, Structure
  from pymatgen.io.cif import CifWriter
except Exception:
  Structure = Lattice = CifWriter = None

st.set_page_config(layout="wide", page_title="Ionic Conductivity Predictor")

# ----------------------------------
# MP 官方配色（可扩充）
# ----------------------------------
MP_COLORS = {
    "H": "#FFFFFF",
    "Li": "#CC80FF",
    "Be": "#C2FF00",
    "B": "#FFB5B5",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Na": "#AB5CF2",
    "Mg": "#8AFF00",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "K": "#8F40D4",
    "Ca": "#FFD478",
    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Ga": "#C28F8F",
    "Ge": "#4C4CFF",
    "As": "#BD80E3",
    "Se": "#FFA100",
    "Br": "#A62929",
    "Kr": "#5CB8D1",
    "Rb": "#702EB0",
    "Sr": "#00FF00",
    "Y": "#94FFFF",
    "Zr": "#94E0E0",
    "Nb": "#73C2C9",
    "Mo": "#54B5B5",
    "Ru": "#248F8F",
    "Rh": "#0A7D8C",
    "Pd": "#006985",
    "Ag": "#C0C0C0",
    "Cd": "#FFD98F",
    "In": "#A67573",
    "Sn": "#668080",
    "Sb": "#9E63B5",
    "Te": "#D47A00",
    "I": "#940094",
    "Xe": "#4DC4FF",
    "Cs": "#57178F",
    "Ba": "#00C900",
    "La": "#70D4FF",
    "Ce": "#FFFFC7",
    "Pr": "#D9FFC7",
    "Nd": "#C7FFC7",
    "Pm": "#A3FFC7",
    "Sm": "#8FFFC7",
    "Eu": "#61FFC7",
    "Gd": "#45FFC7",
    "Tb": "#30FFC7",
    "Dy": "#1FFFC7",
    "Ho": "#00FF9C",
    "Er": "#00E675",
    "Tm": "#00D452",
    "Yb": "#00BF69",
    "Lu": "#00AB6B",
    "Hf": "#4DC2FF",
    "Ta": "#4DA6FF",
    "W": "#2194D6",
    "Re": "#267DAB",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#D0D0E0",
    "Au": "#FFD123",
    "Hg": "#B8B8D0",
    "Tl": "#A6544D",
    "Pb": "#575961",
    "Bi": "#9E4FB5",
    "Po": "#AB5C00",
    "At": "#754F45",
    "Rn": "#428296",
    "Fr": "#420066",
    "Ra": "#00C900",
    "Ac": "#70ABFA",
    "Th": "#00BAFF",
    "Pa": "#00A1FF",
    "U": "#008FFF",
    "Np": "#0080FF",
    "Pu": "#006BFF",
}

# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 45%;
        background-color: #f9f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }
    .rounded-container h2 {
        margin-top: -80px;
        text-align: center;
        background-color: #e0e0e0e0;
        padding: 10px;
        border-radius: 10px;
    }
    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 10px;
        font-size: 1.1em;
        border-radius: 10px;
    }
    .stMetric {
        font-size: 0.9em;
    }
    .stWrite {
        font-size: 0.9em;
    }
    h3 {
        font-size: 1.2em;
        margin-bottom: 0.5em;
    }
    .dataframe {
        font-size: 0.8em;
    }
    div[data-testid="column"] {
        padding: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2 style="font-size:24px;"> Predict Ionic Conductivity of Solid Electrolytes</h2>
        <blockquote>
            1. This web app predicts ionic conductivity of solid electrolytes based on material composition features.<br>
             2. Enter a valid chemical formula string below to get the predicted result. (Note: Restricted to inorganic crystalline solid electrolytes).
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- Inputs --------------------
col1, col2 = st.columns([2, 1])
with col1:
  formula_input = st.text_input(
      "Enter Chemical Formula:", placeholder="e.g., Li7La3Zr2O12"
  )
  temperature = st.number_input(
      "Temperature (K):", min_value=200, max_value=1000, value=298, step=1
  )
  submit_button = st.button("Submit and Predict")
with col2:
  MP_API_KEY_DEFAULT = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"
  mp_key_input = st.text_input(
      "Materials Project API key :", type="password", value=MP_API_KEY_DEFAULT
  )
  use_placeholder_checkbox = st.checkbox(
      "Always use placeholder structure (ignore MP)", value=False
  )
  st.markdown("If MP key left empty, app will use placeholder cell.")

# 指定的描述符列表 - 七个特征
required_descriptors = [
    "MagpieData mean CovalentRadius",
    "Temp",
    "MagpieData avg_dev SpaceGroupNumber",
    "0-norm",
    "MagpieData mean MeltingT",
    "MagpieData avg_dev Column",
    "MagpieData mean NValence",
]


@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
  """缓存模型加载，避免重复加载导致内存溢出"""
  # 请确保此处的模型文件夹名称与你实际训练后保存的文件夹一致
  return TabularPredictor.load("./ag-20251024_075719")


# 材料特征计算函数
def calculate_material_features(formula):
  """计算材料的组成特征"""
  try:
    from matminer.featurizers.composition import (
        ElementProperty,
        IonProperty,
        Meredig,
        Stoichiometry,
    )
    from matminer.featurizers.conversions import (
        CompositionToOxidComposition,
        StrToComposition,
    )

    df = pd.DataFrame({"Formula": [formula]})
    stc = StrToComposition()
    df = stc.featurize_dataframe(df, "Formula", ignore_errors=True)

    if "composition" not in df.columns or df["composition"].iloc[0] is None:
      return {"Formula": formula}

    features = {"Formula": formula}

    ep = ElementProperty.from_preset("magpie")
    df = ep.featurize_dataframe(df, "composition", ignore_errors=True)

    mer = Meredig()
    df = mer.featurize_dataframe(df, "composition", ignore_errors=True)

    sto = Stoichiometry()
    df = sto.featurize_dataframe(df, "composition", ignore_errors=True)

    try:
      cto = CompositionToOxidComposition()
      df = cto.featurize_dataframe(df, "composition_oxid", ignore_errors=True)
      if "composition_oxid" in df.columns:
        ion = IonProperty()
        df = ion.featurize_dataframe(df, "composition_oxid", ignore_errors=True)
    except Exception:
      pass

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
      val = df[col].iloc[0]
      features[col] = float(val) if not pd.isna(val) else 0.0

    return features

  except Exception as e:
    st.warning(f"Feature calculation failed: {e}")
    return {"Formula": formula}


def filter_selected_features(features_dict, selected_descriptors, temperature):
  filtered_features = {}
  filtered_features["Temp"] = float(temperature)

  for feature_name in selected_descriptors:
    if feature_name == "Temp":
      continue
    if feature_name in features_dict:
      filtered_features[feature_name] = features_dict[feature_name]
    else:
      filtered_features[feature_name] = 0.0
  return filtered_features


# ------------------------------- MP structure loader -------------------------------
def load_structure_from_mp(formula, api_key):
  if MPRester is None:
    return None, "mp-api not installed"
  try:
    with MPRester(api_key) as mpr:
      results = mpr.summary.search(formula=formula, fields=["structure"])
      if not results:
        return None, "No MP entry found"
      doc = results[0]
      try:
        struct = doc.structure.get_primitive_structure()
      except Exception:
        struct = doc.structure
      return struct, "Successfully loaded from MP"
  except Exception as e:
    return None, f"MP error: {e}"


def generate_placeholder_structure(formula):
  elems = re.findall(r"[A-Z][a-z]?", formula or "")
  elems = list(dict.fromkeys(elems))
  if len(elems) == 0:
    elems = ["Li", "O"]
  coords = []
  n = len(elems)
  for i in range(n):
    coords.append([
        0.1 + 0.8 * ((i + 1) / (n + 1)),
        0.1 + 0.6 * random.random(),
        0.1 + 0.6 * random.random(),
    ])
  if Lattice is None or Structure is None:
    st.warning("pymatgen not installed — cannot create placeholder Structure.")
    return None
  lattice = Lattice.cubic(10.0)
  struct = Structure(lattice, elems, coords)
  return struct


def structure_to_cif_string(structure):
  if CifWriter is None:
    st.warning("pymatgen.io.cif.CifWriter not available.")
    return None
  tmp = None
  try:
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
      fname = tmp.name
    try:
      CifWriter(structure).write_file(fname)
    except Exception as e:
      try:
        structure.to(filename=fname)
      except Exception:
        raise e
    with open(fname, "r", encoding="utf-8") as f:
      cif_str = f.read()
    return cif_str
  finally:
    try:
      if tmp is not None:
        os.unlink(tmp.name)
    except Exception:
      pass


def render_structure_with_legend(structure, width=520, height=260):
  cif_str = structure_to_cif_string(structure)
  if not cif_str:
    return None

  view = py3Dmol.view(width=360, height=height)
  view.addModel(cif_str, "cif")

  for i, site in enumerate(structure.sites):
    el = str(site.specie)
    color = MP_COLORS.get(el, "#9E9E9E")
    view.setStyle(
        {"index": i},
        {"sphere": {"radius": 0.45, "color": color}, "stick": {"radius": 0.18, "color": color}},
    )

  view.addUnitCell()
  view.zoomTo()
  structure_html = view._make_html()

  elements = sorted({str(s.specie) for s in structure.sites})
  legend_items = ""
  for el in elements:
    c = MP_COLORS.get(el, "#9E9E9E")
    legend_items += f"""
        <div style="display:flex;align-items:center;margin-bottom:6px;">
            <div style="
                width:14px;
                height:14px;
                background:{c};
                border:1px solid #333;
                border-radius:3px;
                margin-right:6px;
            "></div>
            <span style="font-size:13px;color:#222;">{el}</span>
        </div>
        """

  legend_html = f"""
    <div style="
        background:#f5f5f5;
        border:1px solid #ccc;
        border-radius:8px;
        padding:10px;
        width:120px;
    ">
        <div style="text-align:center;font-weight:600;margin-bottom:8px;">
            Element colors
        </div>
        {legend_items}
    </div>
    """

  final_html = f"""
    <div style="
        display:flex;
        align-items:flex-start;
        gap:12px;
        width:{width}px;
    ">
        <div>{structure_html}</div>
        {legend_html}
    </div>
    """
  return final_html


# ------------------------------- 主要执行部分 -------------------------------
if submit_button:
  if not formula_input:
    st.error("Please enter a formula.")
    st.stop()

  with st.spinner("Processing..."):
    structure = None
    mp_id = None
    mp_msg = ""

    if (mp_key_input and not use_placeholder_checkbox) and (
        MPRester is not None
    ):
      try:
        struct, info = load_structure_from_mp(formula_input, mp_key_input)
        if struct:
          structure = struct
          mp_id = info
          mp_msg = f"Loaded from Materials Project: {mp_id}"
        else:
          mp_msg = f"MP lookup failed: {info}"
      except Exception as e:
        mp_msg = f"MP lookup exception: {e}"
    else:
      if use_placeholder_checkbox:
        mp_msg = "Placeholder structure selected by user."
      else:
        mp_msg = "MP key not provided or mp-api not installed."

    if mp_msg and not mp_msg.startswith("Loaded from Materials Project"):
      st.warning(f"⚠️ Structure Warning: {mp_msg}")
    if structure is None:
      structure = generate_placeholder_structure(formula_input)
      if structure is None:
        st.error("Could not generate any structure (pymatgen missing).")
        st.stop()

    st.subheader("Crystal Structure Preview (Unit Cell)")
    html = render_structure_with_legend(structure)

    if html:
      components.html(html, height=280, scrolling=False)
    else:
      st.error("Failed to render structure.")

    features = calculate_material_features(formula_input)
    selected_features = filter_selected_features(
        features, required_descriptors, temperature
    )
    feature_df = pd.DataFrame([selected_features])

    st.subheader("Material Features")
    st.dataframe(feature_df)

    if features:
      input_data = {
          "Formula": [formula_input],
          "Temp": [temperature],
      }

      numeric_features = {}
      for feature_name in required_descriptors:
        if feature_name == "Temp":
          numeric_features[feature_name] = [temperature]
        elif feature_name in features:
          numeric_features[feature_name] = [features[feature_name]]
        else:
          numeric_features[feature_name] = [0.0]

      input_data.update(numeric_features)
      input_df = pd.DataFrame(input_data)

      try:
        predictor = load_predictor()

        # 【核心修改点 1】：去掉了 KNeighborsDist 模型，仅保留 5 个高性能核心模型
        essential_models = [
            "CatBoost",
            "ExtraTreesMSE",
            "LightGBM",
            "WeightedEnsemble_L2",
            "XGBoost",
        ]

        predict_df = input_df.copy()
        predictions_dict = {}
        base_preds = []

        for model in essential_models:
          try:
            predictions = predictor.predict(predict_df, model=model)
            val = float(predictions.iloc[0])
            predictions_dict[model] = [val]
            if model != "WeightedEnsemble_L2":
              base_preds.append(val)
          except Exception as model_error:
            st.warning(f"Model {model} prediction failed: {str(model_error)}")
            predictions_dict[model] = ["Error"]

        st.write("Prediction Results (Essential Models):")
        results_df = pd.DataFrame(predictions_dict)
        st.dataframe(results_df)

        # 【核心修改点 2】：增加预测不确定性指标（计算基模型输出的离散度/标准差）
        if base_preds:
          uncertainty_std = np.std(base_preds)
          st.info(
              f"📊 **Prediction Uncertainty Indicator (Base-Learner Disagreement):**"
              f" **{uncertainty_std:.4f}** (Std. Dev.)\n\n"
              "*Note: This value quantifies the dispersion among the constituent"
              " base models (CatBoost, ExtraTreesMSE, LightGBM, and"
              " XGBoost). A higher disagreement indicates that the queried"
              " composition approaches the boundaries of the model's"
              " applicability domain and should be interpreted with caution.*"
          )

      except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
