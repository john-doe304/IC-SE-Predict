import importlib, sys
import numpy as _np
if not hasattr(_np, "product"):
    _np.product = _np.prod
import streamlit as st
import os
import gc
import re
import requests
import numpy as np
import pandas as pd
import py3Dmol
from io import BytesIO
from autogluon.tabular import TabularPredictor

# --- RDKit ---
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DSVG

# --- Matminer ---
from mordred import Calculator, descriptors
from matminer.featurizers.composition import (
    ElementProperty, Meredig, Stoichiometry, IonProperty
)
from matminer.featurizers.conversions import (
    StrToComposition, CompositionToOxidComposition
)


# --- Pymatgen ---
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition

# =====================================================
#  Materials Project API KEY
# =====================================================
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"


# =====================================================
# 自定义颜色和价态映射 (匹配 Materials Project 网站的 LLZO 样式)
# =====================================================
CUSTOM_LEGEND_COLORS = {
    "Li": "#90EE90",  # 浅绿色 (Light Green)
    "La": "#3CB371",  # 中绿色 (Medium Sea Green)
    "Zr": "#00FF00",  # 亮绿色 (Lime)
    "O": "#FF0000",   # 红色 (Red)
    "P": "#FFA500",   # 磷的颜色示例
    "S": "#FFD700",   # 硫的颜色示例
    "Ge": "#668F8F",  # 锗的颜色示例
    "Cl": "#00FF00",  # 氯的颜色示例 (与 Zr 冲突，但作为示例)
    "Y": "#80FFB3",   # 钇的颜色示例
}

# 辅助函数：将价态数字转换为 HTML 格式的上下标字符串
def format_charge(charge):
    if charge > 0:
        return f"+{int(charge)}"
    elif charge < 0:
        return f"{int(charge)}"
    return ""


# =====================================================
# Streamlit 页面样式
# =====================================================
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='rounded-container'>
        <h2 style="font-size:24px;"> Predict Ionic Conductivity of Solid Electrolytes</h2>
        <blockquote>
            1. This web app predicts ionic conductivity of solid electrolytes based on material composition features.<br>
            2. Enter a valid chemical formula string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 输入区
# =====================================================
formula_input = st.text_input(
    "Enter Chemical Formula of the Material:",
    placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6",
)

temperature = st.number_input(
    "Select Temperature (K):",
    min_value=200,
    max_value=1000,
    value=298,
    step=10,
)

submit_button = st.button("Submit and Predict")


# =====================================================
# 缓存模型加载
# =====================================================
@st.cache_resource(show_spinner=False)
def load_predictor():
    # 假设模型文件 ag-20251024_075719 存在于当前目录下
    return TabularPredictor.load("./ag-20251024_075719")


# =====================================================
# 1. Public database structure retrieval
# =====================================================
def load_from_MP(formula: str):
    """
    Extremely robust MP structure loader:
    - No occupancy filtering (avoids occu errors)
    - Always returns MP's conventional standard structure (same as website)
    - Never touches partial occupancy atoms
    - Guarantees no 'occu' errors
    """

    try:
        with MPRester(MP_API_KEY) as mpr:

            # 1. summary.search: modern API
            try:
                results = mpr.summary.search(formula=formula)
                if results:
                    # pick the FIRST result (not lowest energy!)
                    entry = results[0]

                    # get structure object (pymatgen.core.Structure)
                    s = entry.structure

                    # convert to conventional standard structure
                    try:
                        s = s.get_conventional_structure()
                    except:
                        pass

                    return s
            except Exception:
                pass

            # 2. fallback: query
            try:
                q = mpr.query(criteria={"formula": formula}, properties=["material_id"])
                if q:
                    mid = q[0]["material_id"]
                    s = mpr.get_structure_by_material_id(mid)

                    try:
                        s = s.get_conventional_structure()
                    except:
                        pass

                    return s
            except Exception:
                pass

            # 3. fallback: entries
            try:
                es = mpr.get_entries(formula)
                if es:
                    s = es[0].structure
                    try:
                        s = s.get_conventional_structure()
                    except:
                        pass
                    return s
            except Exception:
                pass

            # 4. fallback: get_structures
            try:
                ss = mpr.get_structures(formula)
                if ss:
                    s = ss[0]
                    try:
                        s = s.get_conventional_structure()
                    except:
                        pass
                    return s
            except Exception:
                pass

        return None

    except Exception as e:
        st.error(f"Materials Project fetch failed: {e}")
        return None



def load_from_COD(formula):
    """Fallback database"""
    try:
        url = f"https://www.crystallography.net/cod/result?format=core-formula&q={formula}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        lines = r.text.strip().split()
        if len(lines) == 0:
            return None

        cod_id = lines[0]
        cif_url = f"https://www.crystallography.net/cod/{cod_id}.cif"
        cif_bytes = requests.get(cif_url).content

        return Structure.from_str(cif_bytes.decode(), fmt="cif")
    except:
        return None


def load_crystal_structure_public(formula):
    st.info("Searching public databases for crystal structure...")

    s = load_from_MP(formula)
    if s:
        st.success("Structure found in Materials Project ✓")

        # -------- ★ 关键：标准化为 Materials Project 传统晶胞（官网也用它） --------
        try:
            s = s.get_conventional_structure()
            s = s.as_dict()  # 防止 py3Dmol 读取错误
            s = Structure.from_dict(s)
        except:
            pass

        return s

    s = load_from_COD(formula)
    if s:
        st.success("Structure found in COD ✓")
        return s

    st.error("No structure found in public databases.")
    return None

def make_supercell(structure, size=(2,2,2)):
    try:
        structure = structure.copy()
        structure.make_supercell(size)
        return structure
    except:
        return structure


# =====================================================
# 2. Structure display (py3Dmol) - 最终版：包含多面体和自定义图例
# =====================================================
def display_structure_py3Dmol(structure):
    # Step 0 - 确保结构是 Conventional Cell
    try:
        structure = structure.get_conventional_structure()
        # 尝试计算氧化态。如果失败，则不显示价态。
        try:
            structure.add_oxidation_state_by_guess()
            oxi_state_dict = structure.composition.oxi_state_dict()
        except:
            oxi_state_dict = {}
            st.warning("Could not determine oxidation states for the legend.")
            
    except Exception as e:
        oxi_state_dict = {}
        st.error(f"Error processing structure for display: {e}")
        return

    # Step 1 - MP style supercell (2x2x2)
    structure_to_display = make_supercell(structure, (2, 2, 2))
    cif_str = structure_to_display.to(fmt="cif")

    # -----------------------------------------------------------------
    # A. 渲染 3D 视图
    # -----------------------------------------------------------------
    # 使用 st.columns 来创建一个更干净的布局容器
    col_3d, col_empty = st.columns([1, 0.01]) 
    
    with col_3d:
        view = py3Dmol.view(width=650, height=520)
        view.addModel(cif_str, "cif")

        # 使用 Jmol 颜色方案渲染原子球和键
        view.setStyle({
            "sphere": {
                "scale": 0.30,
                "colorscheme": "Jmol" 
            },
            "stick": {
                "radius": 0.12,
                "colorscheme": "Jmol"
            }
        })

        # 添加晶胞边界
        view.addUnitCell({"color": "white", "linewidth": 2.0})
        view.setBackgroundColor("white")
        view.setProjection("orthographic")
        view.zoomTo()

        # ★★★ 关键：添加多面体渲染 ★★★
        # 假设常见的中心离子是 La 和 Zr (或结构中原子数最少的阳离子)
        # 这段代码需要根据具体结构调整，这里以 LLZO 为例
        
        # 识别阳离子（除了 Li，因为 Li 通常是移动离子，不作为多面体中心）
        center_elements = [str(el) for el in structure.elements if str(el) not in ['Li', 'O', 'S', 'Cl']]
        
        if not center_elements:
             # 如果没有其他重元素，则尝试 Li 或其他阳离子
             center_elements = [str(el) for el in structure.elements if str(el) in ['Zr', 'La', 'Ge', 'P', 'Y']]

        # 遍历中心元素，以 O 为配位原子渲染多面体
        for center_el in center_elements:
            if center_el in ['Zr', 'La']:
                # 使用自定义的颜色 (例如，为 Zr 使用亮绿，为 La 使用中绿)
                poly_color = CUSTOM_LEGEND_COLORS.get(center_el, "gray")
                
                view.addStyle({"select": f"elem {center_el}"}, {
                    "polyhedra": {
                        "color": poly_color,      
                        "opacity": 0.3,         
                        "hidden": False,        
                        "threshold": 2.5,       # 键长阈值
                        "center": f"elem {center_el}",
                        "vertex": "elem O",     # 假设是氧化物
                        "radius": 0.12          
                    }
                })

        st.components.v1.html(view._make_html(), height=540, scrolling=False)
        
        # -----------------------------------------------------------------
        # B. 手动创建圆角胶囊状图例 (Legend)
        # -----------------------------------------------------------------
        
        # 1. 获取结构中的唯一元素并排序
        elements = [str(e) for e in structure.composition.elements]
        unique_elements = sorted(list(set(elements))) 

        legend_items = []
        
        # 2. 构造图例的 HTML 标记
        for element in unique_elements:
            color = CUSTOM_LEGEND_COLORS.get(element, "#BBBBBB") 
            
            charge_value = oxi_state_dict.get(element, 0)
            charge_text = format_charge(charge_value)
            
            # 构造每个图例项：圆角胶囊样式
            item_html = f"""
            <div style='
                display: flex; 
                align-items: center; 
                justify-content: center;
                margin-left: 10px;
                padding: 8px 15px; 
                background-color: {color}; 
                border-radius: 25px; 
                box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
                min-width: 60px;
                height: 35px;
            '>
                <span style='font-weight: bold; font-size: 1.1em; color: #fff; text-shadow: 1px 1px 1px rgba(0,0,0,0.5);'>
                    {element}<sup>{charge_text}</sup>
                </span>
            </div>
            """
            legend_items.append(item_html)

        # 3. 组合所有图例项，使其水平右对齐显示
        legend_html = f"""
        <div style='
            display: flex; 
            justify-content: flex-end; 
            align-items: center; 
            margin-top: -15px; 
            margin-bottom: 10px;
            width: 100%;
        '>
            {''.join(legend_items)}
        </div>
        """
        
        # 4. 使用 st.markdown 渲染图例
        st.markdown(legend_html, unsafe_allow_html=True)


# =====================================================
# 3. Feature extraction
# =====================================================
def calculate_material_features(formula):
    try:
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
            df, "composition", ignore_errors=True
        )
        df = IonProperty().featurize_dataframe(
            df, "composition_oxid", ignore_errors=True
        )

        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        return {"Formula": formula}


# 指定的 7 个特征
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
    result = {"Temp": float(temperature)}
    for f in selected:
        if f != "Temp":
            result[f] = features.get(f, 0.0)
    return result


# =====================================================
# 4. Main Program Flow
# =====================================================
if submit_button:

    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    # ---------- (1) Load & Show Crystal Structure ----------
    st.subheader("Crystal Structure")

    structure = load_crystal_structure_public(formula_input)

    if structure:
        display_structure_py3Dmol(structure)
    else:
        st.warning("Cannot find structure for this material.")

    # ---------- (2) Feature Extraction ----------
    with st.spinner("Extracting features..."):
        features = calculate_material_features(formula_input)
        st.write(f"Extracted {len(features)} features.")

        selected = filter_selected_features(features, required_descriptors, temperature)
        st.subheader("Selected Features")
        st.dataframe(pd.DataFrame([selected]))

    # ---------- (3) Prediction ----------
    st.subheader("Prediction Results")

    try:
        predictor = load_predictor()
    except:
        predictor = None
        st.error("Model loading failed. Please ensure the model file is present.")

    if predictor:
        input_data = {"Formula": [formula_input], "Temp": [temperature]}
        for f in required_descriptors:
            if f != "Temp":
                input_data[f] = [features.get(f, 0.0)]
        input_df = pd.DataFrame(input_data)

        models = [
            "CatBoost",
            "ExtraTreesMSE",
            "LightGBM",
            "KNeighborsDist",
            "WeightedEnsemble_L2",
            "XGBoost",
        ]

        results = {}
        for model in models:
            try:
                # 确保输入数据的列与训练模型时使用的特征列一致
                # 这里假设 required_descriptors 包含了训练时需要的全部特征
                predict_df = input_df[required_descriptors].copy() 
                results[model] = predictor.predict(predict_df, model=model)
            except Exception as e:
                results[model] = f"Error: {e}"

        st.dataframe(pd.DataFrame(results).iloc[:1, :])

        del predictor
        gc.collect()
