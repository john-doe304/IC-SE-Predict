#############################################################
#  修复 numpy "product" for Streamlit Cloud
#############################################################
import numpy as _np
if not hasattr(_np, "product"):
    _np.product = _np.prod


#############################################################
#  Imports
#############################################################
import streamlit as st
import gc
import requests
import numpy as np
import pandas as pd
import py3Dmol

from autogluon.tabular import TabularPredictor
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.ext.matproj import MPRester


#############################################################
#  Materials Project API Key
#############################################################
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"


#############################################################
#  Jmol Color Scheme (用于右下角 legend)
#############################################################
Jmol_colors = {
    "H": "#FFFFFF",
    "Li": "#CC80FF",
    "La": "#FFCE00",
    "Zr": "#94FFFF",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "P": "#FF8000",
    "Ge": "#668F8F",
    # 默认颜色
    "DEFAULT": "#A0A0A0"
}


#############################################################
#  Streamlit Style
#############################################################
st.markdown("""
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 40px auto;
        max-width: 45%;
        padding: 20px;
        background-color: #FAFAFA;
    }
    </style>
""", unsafe_allow_html=True)


#############################################################
#  MP structure loader
#############################################################
def load_from_MP(formula):
    try:
        with MPRester(MP_API_KEY) as mpr:
            comp = Composition(formula).reduced_formula
            results = mpr.summary.search(formula=comp)

            if results:
                results.sort(key=lambda x: x.energy_per_atom)
                s = results[0].structure
                try:
                    s = s.get_conventional_structure()
                except:
                    pass
                return s
            else:
                return None

    except Exception as e:
        st.error(f"MP fetch error: {e}")
        return None


#############################################################
#  Display structure in py3Dmol (fast, stable)
#############################################################
def display_structure_py3Dmol(structure):

    structure = structure.copy()
    cif = structure.to(fmt="cif")

    view = py3Dmol.view(width=650, height=520)
    view.addModel(cif, "cif")

    view.setStyle({
        "stick": {"radius": 0.16},
        "sphere": {"scale": 0.30, "colorscheme": "Jmol"}
    })

    view.addUnitCell({"color": "white", "linewidth": 2})
    view.setBackgroundColor("white")
    view.setProjection("orthographic")
    view.zoomTo()

    st.components.v1.html(view._make_html(), height=540, scrolling=False)


#############################################################
#  Show element color legend (右下角)
#############################################################
def display_color_legend(structure):

    elements = sorted(structure.symbol_set)

    legend_html = """
        <div style='position: relative; text-align: right;'>
        <div style='display: inline-block; padding: 10px; border: 1px solid #ccc; border-radius: 10px; background: #FFFFFF;'>
            <b>Element Colors (Jmol)</b><br>
    """

    for el in elements:
        color = Jmol_colors.get(el, Jmol_colors["DEFAULT"])
        legend_html += f"""
            <div style="margin-top:4px;">
                <span style="
                    display:inline-block;
                    width:14px;
                    height:14px;
                    background:{color};
                    border-radius:50%;
                    margin-right:6px;
                    border: 1px solid #555;
                "></span>
                <span style="font-size:14px;">{el}</span>
            </div>
        """

    legend_html += "</div></div>"

    st.markdown(legend_html, unsafe_allow_html=True)



#############################################################
#  Feature extraction
#############################################################
def calculate_material_features(formula):
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Meredig, Stoichiometry, IonProperty
        )
        from matminer.featurizers.conversions import (
            StrToComposition, CompositionToOxidComposition
        )

        df = pd.DataFrame({"Formula": [formula]})
        df = StrToComposition().featurize_dataframe(df, "Formula", ignore_errors=True)

        if "composition" not in df.columns:
            return {"Formula": formula}

        features = {"Formula": formula}

        ep = ElementProperty.from_preset("magpie")
        df = ep.featurize_dataframe(df, "composition", ignore_errors=True)

        df = Meredig().featurize_dataframe(df, "composition", ignore_errors=True)
        df = Stoichiometry().featurize_dataframe(df, "composition", ignore_errors=True)

        df = CompositionToOxidComposition().featurize_dataframe(
            df, "composition", ignore_errors=True)
        df = IonProperty().featurize_dataframe(
            df, "composition_oxid", ignore_errors=True)

        for col in df.select_dtypes(include=[np.number]).columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature extraction error: {e}")
        return {"Formula": formula}


required_descriptors = [
    "MagpieData mean CovalentRadius",
    "Temp",
    "MagpieData avg_dev SpaceGroupNumber",
    "0-norm",
    "MagpieData mean MeltingT",
    "MagpieData avg_dev Column",
    "MagpieData mean NValence",
]


def filter_selected_features(features, selected, temperature):
    out = {"Temp": float(temperature)}
    for f in selected:
        if f != "Temp":
            out[f] = features.get(f, 0.0)
    return out


#############################################################
#  UI INPUT
#############################################################
st.title("Predict Ionic Conductivity of Solid Electrolytes")

formula = st.text_input("Chemical Formula", "Li7La3Zr2O12")
temperature = st.number_input("Temperature (K)", 200, 1000, 298)

submit = st.button("Submit & Predict")


#############################################################
#  MAIN FLOW
#############################################################
if submit:

    st.subheader("Crystal Structure")

    s = load_from_MP(formula)

    if s:
        display_structure_py3Dmol(s)
        display_color_legend(s)   # 🔥 在右下角显示颜色 legend
    else:
        st.error("No structure found in Materials Project.")
        st.stop()

    # =============== FEATURE EXTRACTION ===============
    with st.spinner("Extracting features..."):
        feats = calculate_material_features(formula)
        feats_sel = filter_selected_features(feats, required_descriptors, temperature)
        st.subheader("Selected Features")
        st.dataframe(pd.DataFrame([feats_sel]))

    # =============== PREDICTION ===============
    predictor = TabularPredictor.load("./ag-20251024_075719")

    df_in = {"Formula": [formula], "Temp": [temperature]}
    for f in required_descriptors:
        if f != "Temp":
            df_in[f] = [feats.get(f, 0.0)]
    df_in = pd.DataFrame(df_in)

    models = [
        "CatBoost",
        "ExtraTreesMSE",
        "LightGBM",
        "KNeighborsDist",
        "WeightedEnsemble_L2",
        "XGBoost",
    ]

    out = {}

    for m in models:
        try:
            out[m] = predictor.predict(df_in, model=m)
        except:
            out[m] = "Error"

    st.subheader("Prediction Results")
    st.dataframe(pd.DataFrame(out).iloc[:1, :])


#############################################################
# END
#############################################################
