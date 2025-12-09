# -----------------------------------------------------------
#   Solid Electrolyte Ionic Conductivity Predictor (Final)
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
    # rdkit optional; warn at runtime if missing when used
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
    /* 减小指标卡片的字体大小 */
    .stMetric {
        font-size: 0.9em;
    }
    /* 减小特征提取成功信息的字体大小 */
    .stWrite {
        font-size: 0.9em;
    }
    /* 减小子标题的字体大小 */
    h3 {
        font-size: 1.2em;
    }
    /* 减小数据框的字体大小 */
    .dataframe {
        font-size: 0.8em;
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
             2.  Enter a valid chemical formula string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- Inputs --------------------
col1, col2 = st.columns([2, 1])
with col1:
    formula_input = st.text_input("Enter Chemical Formula:", placeholder="e.g., Li7La3Zr2O12")
    temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=1)
    submit_button = st.button("Submit and Predict")
with col2:
    MP_API_KEY_DEFAULT = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"
    mp_key_input = st.text_input("Materials Project API key :", type="password", value=MP_API_KEY_DEFAULT)
    use_placeholder_checkbox = st.checkbox("Always use placeholder structure (ignore MP)", value=False)
    st.markdown("If MP key left empty, app will use placeholder cell.")

# 指定的描述符列表 - 你选择的七个特征
required_descriptors = [
    'MagpieData mean CovalentRadius',
    'Temp',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence'
]

# 缓存模型加载器以避免重复加载
@st.cache_resource(show_spinner=False, max_entries=1)  # 限制只缓存一个实例
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load("./ag-20251024_075719")

def mol_to_image(mol, size=(200, 200)):
    """将分子转换为背景颜色为 #f9f9f9f9 的SVG图像"""
    # 创建绘图对象
    d2d = MolDraw2DSVG(size[0], size[1])
    
    # 获取绘图选项
    draw_options = d2d.drawOptions()
    
    # 设置背景颜色为 #f9f9f9f9
    draw_options.background = '#f9f9f9'
    
    # 移除所有边框和填充
    draw_options.padding = 0.0
    draw_options.additionalBondPadding = 0.0
    
    # 移除原子标签的边框
    draw_options.annotationFontScale = 1.0
    draw_options.addAtomIndices = False
    draw_options.addStereoAnnotation = False
    draw_options.bondLineWidth = 1.5
    
    # 禁用所有边框
    draw_options.includeMetadata = False
    
    # 绘制分子
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    
    # 获取SVG内容
    svg = d2d.GetDrawingText()
    
    # 移除SVG中所有可能存在的边框元素
    # 1. 移除黑色边框矩形
    svg = re.sub(r'<rect [^>]*stroke:black[^>]*>', '', svg, flags=re.DOTALL)
    svg = re.sub(r'<rect [^>]*stroke:#000000[^>]*>', '', svg, flags=re.DOTALL)
    
    # 2. 移除所有空的rect元素
    svg = re.sub(r'<rect[^>]*/>', '', svg, flags=re.DOTALL)
    
    # 3. 确保viewBox正确设置
    if 'viewBox' in svg:
        # 设置新的viewBox以移除边距
        svg = re.sub(r'viewBox="[^"]+"', f'viewBox="0 0 {size[0]} {size[1]}"', svg)
    
    return svg

# 材料特征计算函数
def calculate_material_features(formula):
    """计算材料的组成特征"""
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Meredig, Stoichiometry, IonProperty
        )
        from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

        df = pd.DataFrame({'Formula': [formula]})
        stc = StrToComposition()
        df = stc.featurize_dataframe(df, 'Formula', ignore_errors=True)

        if 'composition' not in df.columns or df['composition'].iloc[0] is None:
            return {'Formula': formula}

        features = {'Formula': formula}

        # 元素属性特征
        ep = ElementProperty.from_preset('magpie')
        df = ep.featurize_dataframe(df, 'composition', ignore_errors=True)

        # Meredig
        mer = Meredig()
        df = mer.featurize_dataframe(df, 'composition', ignore_errors=True)

        # 化学计量特征
        sto = Stoichiometry()
        df = sto.featurize_dataframe(df, 'composition', ignore_errors=True)

        # 离子特征
        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, 'composition', ignore_errors=True)
        ion = IonProperty()
        df = ion.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)

        # 数值特征提取
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return {'Formula': formula}

# 过滤特征 - 只显示指定的七个特征
def filter_selected_features(features_dict, selected_descriptors, temperature):
    """只显示选定的七个特征"""
    filtered_features = {}
    
    # 添加温度特征
    
    filtered_features['Temp'] = float(temperature)
    
    # 添加选定的七个特征
    for feature_name in selected_descriptors:
        if feature_name == 'Temp':
            continue
        
        if feature_name in features_dict:
            filtered_features[feature_name] = features_dict[feature_name]
        else:
            # 如果特征不存在，设为0
            filtered_features[feature_name] = 0.0
    
    return filtered_features

# 自动匹配模型特征
def align_features_with_model(features_dict, predictor, temperature, formula):
    if predictor is None:
        return pd.DataFrame([features_dict])

    try:
        model_features = predictor.feature_metadata.get_features()
    except Exception:
        model_features = []

    aligned = {}
    lower_map = {k.lower(): k for k in features_dict.keys()}

    for feat in model_features:
        f_low = feat.lower()
        if feat in features_dict:
            aligned[feat] = features_dict[feat]
        elif f_low in lower_map:
            aligned[feat] = features_dict[lower_map[f_low]]
        elif f_low in ['temp', 'temperature', 'temperature_k']:
            aligned[feat] = temperature
        elif f_low in ['formula']:
            aligned[feat] = formula
      
        else:
            aligned[feat] = 0.0

    return pd.DataFrame([aligned])

# ------------------------------- MP structure loader -------------------------------
def load_structure_from_mp(formula, api_key):
    if MPRester is None:
        return None, "mp-api not installed"
    try:
        with MPRester(api_key) as mpr:
            results = mpr.summary.search(formula=formula)
            if not results:
                return None, "No MP entry found"
            doc = results[0]

            try:
                # *** 强制单胞 ***
                struct = doc.structure.get_primitive_structure()
            except Exception:
                struct = doc.structure

            return struct, doc.material_id
    except Exception as e:
        return None, f"MP error: {e}"


# ------------------------------- Placeholder cell generator -------------------------------
def generate_placeholder_structure(formula):
    elems = re.findall(r"[A-Z][a-z]?", formula or "")
    elems = list(dict.fromkeys(elems))
    if len(elems) == 0:
        elems = ["Li", "O"]  # fallback
    coords = []
    n = len(elems)
    for i in range(n):
        coords.append([0.1 + 0.8*((i+1)/(n+1)), 0.1 + 0.6*random.random(), 0.1 + 0.6*random.random()])
    if Lattice is None or Structure is None:
        st.warning("pymatgen not installed — cannot create placeholder Structure.")
        return None
    lattice = Lattice.cubic(10.0)
    struct = Structure(lattice, elems, coords)
    return struct

# ------------------------------- Structure -> CIF string (robust) -------------------------------
def structure_to_cif_string(structure):
    """Write structure to temp CIF and read string back — robust across pymatgen versions."""
    if CifWriter is None:
        st.warning("pymatgen.io.cif.CifWriter not available.")
        return None
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            fname = tmp.name
        # write
        try:
            CifWriter(structure).write_file(fname)
        except Exception as e:
            # older/newer api differences — try alternate approach
            try:
                # If structure has to() with filename accepted
                structure.to(filename=fname)
            except Exception:
                raise e
        # read back
        with open(fname, "r", encoding="utf-8") as f:
            cif_str = f.read()
        return cif_str
    finally:
        # cleanup
        try:
            if tmp is not None:
                os.unlink(tmp.name)
        except Exception:
            pass

# ------------------------------- Render structure to HTML for Streamlit -------------------------------
def render_structure_to_html(structure, width=300, height=320):
    """Render pymatgen Structure to py3Dmol HTML using atom index (correct)."""
    if structure is None:
        return None

    cif_str = structure_to_cif_string(structure)
    if not cif_str:
        return None

    view = py3Dmol.view(width=width, height=height)
    view.addModel(cif_str, "cif")

    # Correct style: use index (0-based), never serial
    for i, site in enumerate(structure.sites):
        el = str(site.specie)
        color = MP_COLORS.get(el, "#9E9E9E")
        view.setStyle({"index": i}, {
            "sphere": {"radius": 0.45, "color": color},
            "stick": {"radius": 0.18, "color": color}
        })

    # Add bonds explicitly to avoid under/over bonding
    view.addUnitCell()
    view.zoomTo()

    # Build HTML
    model_html = view._make_html()

    # Legend
    elements = sorted({str(s.specie) for s in structure.sites})
    legend_items = ""
    for el in elements:
        c = MP_COLORS.get(el, "#9E9E9E")
        legend_items += f"""
        <div style="display:flex;align-items:center;margin-bottom:4px;">
            <div style="width:14px;height:14px;background:{c};
                border:1px solid #444;border-radius:3px;margin-right:6px;"></div>
            {el}
        </div>
        """

    legend_html = f"""
    <div style="
        position:absolute; bottom:10px; right:10px;
        background:rgba(200,200,200,0.95);
        padding:8px; border-radius:6px;
        border:1px solid #ccc; font-size:12px;
    ">
        <b>Element colors</b><br>{legend_items}
    </div>
    """

    final = f"""
    <div style="position:relative;width:{width}px;height:{height}px;">
        {model_html}
        {legend_html}
    </div>
    """
    return final


# ------------------------------- Main Execution -------------------------------
if submit_button:
    if not formula_input:
        st.error("Please enter a formula.")
        st.stop()

    with st.spinner("Processing..."):
        # decide structure source
        structure = None
        mp_id = None
        mp_msg = ""

        if (mp_key_input and not use_placeholder_checkbox) and (MPRester is not None):
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

        # fallback to placeholder if needed
        if structure is None:
            structure = generate_placeholder_structure(formula_input)
            if structure is None:
                st.error("Could not generate any structure (pymatgen missing).")
                st.stop()

        # render and show HTML in Streamlit using components
        st.subheader("Crystal Structure Preview (Unit Cell)")
       #if mp_id:
          #st.success(mp_msg)
       #else:
           #st.info(mp_msg)

        html = render_structure_to_html(structure, width=400, height=220)
        if html:
            components.html(html, height=260, scrolling=False)
        else:
            st.error("Failed to render structure.")

        # Draw legend separately too (redundant but helpful)
        # draw_element_legend(structure)
        # Features & prediction
        #st.subheader("Extracted Features & Prediction")

        # 计算材料特征
        features = calculate_material_features(formula_input)
        #st.write(f"✅ Total features extracted: {len(features)}")
                
        # 只显示选定的七个特征
        selected_features = filter_selected_features(features, required_descriptors, temperature)
        feature_df = pd.DataFrame([selected_features])
                
        st.subheader("Material Features")
        st.dataframe(feature_df)
            
        if features:
            # 创建输入数据
            input_data = {
                "Formula": [formula_input],
                       
                "Temp": [temperature],
               }
                    
            # 添加数值特征
            numeric_features = {}
            for feature_name in required_descriptors:
                if feature_name == 'Temp':
                    numeric_features[feature_name] = [temperature]
                elif feature_name in features:
                    numeric_features[feature_name] = [features[feature_name]]
                else:
                    numeric_features[feature_name] = [0.0]  # 默认值
                        
            input_data.update(numeric_features)
                        
            input_df = pd.DataFrame(input_data)
                
            # 加载模型并预测
            try:
                # 使用缓存的模型加载方式
                predictor = load_predictor()
                    
                # 只使用最关键的模型进行预测，减少内存占用
                essential_models = ['CatBoost',
                                    'ExtraTreesMSE',
                                    'LightGBM',
                                    'KNeighborsDist',
                                    'WeightedEnsemble_L2',
                                    'XGBoost']
                                        
                predict_df = input_df.copy()
                predictions_dict = {}
                    
                for model in essential_models:
                    try:
                        predictions = predictor.predict(predict_df, model=model)
                        predictions_dict[model] = predictions
                    except Exception as model_error:
                        st.warning(f"Model {model} prediction failed: {str(model_error)}")
                        predictions_dict[model] = "Error"

                # 显示预测结果
                st.write("Prediction Results (Essential Models):")
                st.markdown(
                    "**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                results_df = pd.DataFrame(predictions_dict)
                st.dataframe(results_df.iloc[:1,:])

               # 主动释放内存
               #del predictor
               #gc.collect()

            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")



















