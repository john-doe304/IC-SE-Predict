import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile
import base64
from io import BytesIO
from autogluon.tabular import FeatureMetadata
import gc  # 添加垃圾回收模块
import re  # 添加正则表达式模块用于处理SVG
from tqdm import tqdm 
import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.ext.matproj import MPRester
import plotly.graph_objects as go
import io

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
    /* 晶体结构显示样式 */
    .crystal-structure {
        margin: 20px 0;
        text-align: center;
    }
    .error-message {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
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

# FORMULA 输入区域
formula_input = st.text_input("Enter Chemical Formula of the Material:",placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6", )

# 温度输入
temperature = st.number_input("Select Temperature (K):", min_value=200, max_value=1000, value=298, step=10)

# Materials Project API 密钥输入
mp_api_key = st.text_input("Materials Project API Key (optional):", 
                          placeholder="Enter your API key to view crystal structure",
                          type="password",
                          value="")

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

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
@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load("./ag-20251024_075719")

def validate_chemical_formula(formula):
    """验证化学公式格式"""
    if not formula:
        return False, "Formula cannot be empty"
    
    invalid_chars = set('!@#$%^&*()_+=[]{}|;:,<>?`~')
    if any(char in formula for char in invalid_chars):
        return False, "Formula contains invalid characters"
    
    if not any(c.isalpha() for c in formula):
        return False, "Formula must contain chemical elements"
    
    return True, "Valid formula"

# 常见固态电解质的已知结构信息
KNOWN_STRUCTURES = {
    'Li7La3Zr2O12': {
        'common_name': 'LLZO (Garnet)',
        'space_groups': [230, 225],  # 立方相的空间群
        'typical_phases': ['cubic', 'tetragonal']
    },
    'Li10GeP2S12': {
        'common_name': 'LGPS',
        'space_groups': [15],
        'typical_phases': ['orthorhombic']
    },
    'Li3YCl6': {
        'common_name': 'LYC',
        'space_groups': [166],
        'typical_phases': ['hexagonal']
    },
    'Li6PS5Cl': {
        'common_name': 'Argyrodite',
        'space_groups': [216],
        'typical_phases': ['cubic']
    }
}

def get_best_materials_project_structure(formula, api_key):
    """从Materials Project获取最相关的晶体结构信息"""
    if not api_key or not api_key.strip():
        return None, "No API key provided"
    
    try:
        api_key = api_key.strip()
        
        if len(api_key) != 32 or not all(c.isalnum() for c in api_key):
            return None, "Invalid API key format. API key should be 32 alphanumeric characters."
        
        with MPRester(api_key) as mpr:
            # 获取所有匹配的材料
            materials = mpr.get_entries(formula, inc_structure=True, property_data=["spacegroup", "density", "volume", "formation_energy_per_atom", "band_gap"])
            
            if not materials:
                return None, f"No materials found for formula: {formula}"
            
            # 如果有已知结构信息，优先选择
            if formula in KNOWN_STRUCTURES:
                known_info = KNOWN_STRUCTURES[formula]
                st.info(f"🔍 Searching for {known_info['common_name']} structure...")
                
                # 优先选择已知空间群的结构
                preferred_materials = []
                other_materials = []
                
                for material in materials:
                    spacegroup = getattr(material, 'data', {}).get('spacegroup', {})
                    sg_number = spacegroup.get('number', 0)
                    
                    if sg_number in known_info['space_groups']:
                        preferred_materials.append(material)
                    else:
                        other_materials.append(material)
                
                # 优先选择已知空间群的结构
                if preferred_materials:
                    materials = preferred_materials + other_materials
            
            # 选择最稳定的结构（最低形成能）
            materials.sort(key=lambda x: x.energy_per_atom)
            best_material = materials[0]
            
            structure = best_material.structure
            material_id = best_material.entry_id
            
            # 获取详细数据
            material_data = best_material.data or {}
            spacegroup = material_data.get('spacegroup', {})
            
            # 分析结构特征
            structure_info = analyze_structure_features(structure)
            
            result = {
                'structure': structure,
                'material_id': material_id,
                'spacegroup': spacegroup,
                'density': material_data.get('density', structure_info['density']),
                'volume': material_data.get('volume', structure.volume),
                'formation_energy_per_atom': best_material.energy_per_atom,
                'band_gap': material_data.get('band_gap', 'N/A'),
                'structure_type': structure_info['structure_type'],
                'symmetry': structure_info['symmetry'],
                'formula': formula
            }
            
            return result, None
            
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

def analyze_structure_features(structure):
    """分析晶体结构特征"""
    try:
        # 计算密度
        density = structure.density
        
        # 判断结构类型
        lattice_type = "unknown"
        symmetry = "low"
        
        # 分析晶格参数判断对称性
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        
        # 判断立方晶系
        if abs(a - b) < 0.1 and abs(b - c) < 0.1 and all(abs(angle - 90) < 1 for angle in [alpha, beta, gamma]):
            lattice_type = "cubic"
            symmetry = "high"
        # 判断四方晶系
        elif abs(a - b) < 0.1 and abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 90) < 1:
            lattice_type = "tetragonal"
            symmetry = "medium"
        # 判断六方晶系
        elif abs(a - b) < 0.1 and abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 120) < 1:
            lattice_type = "hexagonal"
            symmetry = "medium"
        else:
            lattice_type = "orthorhombic/triclinic"
            symmetry = "low"
        
        return {
            'density': density,
            'structure_type': lattice_type,
            'symmetry': symmetry
        }
        
    except Exception as e:
        return {
            'density': 'N/A',
            'structure_type': 'unknown',
            'symmetry': 'unknown'
        }

def plot_crystal_structure_plotly(structure, formula):
    """使用plotly绘制3D晶体结构"""
    try:
        # 获取晶格参数
        lattice = structure.lattice
        sites = structure.sites
        
        # 创建原子位置数据
        x, y, z = [], [], []
        colors, sizes, symbols, hover_texts = [], [], [], []
        
        # 原子颜色映射
        color_map = {
            'Li': '#FF6B6B', 'La': '#4ECDC4', 'Zr': '#45B7D1', 'O': '#96CEB4',
            'P': '#FECA57', 'S': '#FF9FF3', 'Cl': '#54A0FF', 'Ge': '#5F27CD',
            'Y': '#00D2D3', 'F': '#FF9F43', 'Br': '#FF7979', 'I': '#BADCF4',
            'Na': '#ABDEE6', 'K': '#D4A5A5', 'Mg': '#FFCC29', 'Ca': '#78C091',
            'Al': '#C7CEEA', 'Si': '#F8C291', 'Ti': '#B8B8B8', 'Fe': '#FF6B6B'
        }
        
        # 原子大小映射
        size_map = {
            'Li': 6, 'La': 12, 'Zr': 10, 'O': 8,
            'P': 9, 'S': 8, 'Cl': 8, 'Ge': 10,
            'Y': 10, 'F': 7, 'Br': 9, 'I': 11,
            'Na': 8, 'K': 10, 'Mg': 9, 'Ca': 10,
            'Al': 9, 'Si': 9, 'Ti': 10, 'Fe': 10
        }
        
        for i, site in enumerate(sites):
            x.append(site.coords[0])
            y.append(site.coords[1])
            z.append(site.coords[2])
            element = site.species_string
            colors.append(color_map.get(element, '#CCCCCC'))
            sizes.append(size_map.get(element, 8))
            symbols.append(element)
            hover_texts.append(f"{element} ({i+1})<br>Position: ({site.coords[0]:.2f}, {site.coords[1]:.2f}, {site.coords[2]:.2f})")
        
        # 创建原子轨迹
        atom_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.9,
                line=dict(width=2, color='darkgray')
            ),
            text=symbols,
            textposition="middle center",
            hoverinfo='text',
            hovertext=hover_texts,
            name='Atoms'
        )
        
        # 创建晶格线
        lines_x, lines_y, lines_z = [], [], []
        
        # 绘制晶格向量
        origin = [0, 0, 0]
        a_vec = lattice.matrix[0]
        b_vec = lattice.matrix[1]
        c_vec = lattice.matrix[2]
        
        # a轴
        lines_x += [origin[0], a_vec[0], None]
        lines_y += [origin[1], a_vec[1], None]
        lines_z += [origin[2], a_vec[2], None]
        
        # b轴
        lines_x += [origin[0], b_vec[0], None]
        lines_y += [origin[1], b_vec[1], None]
        lines_z += [origin[2], b_vec[2], None]
        
        # c轴
        lines_x += [origin[0], c_vec[0], None]
        lines_y += [origin[1], c_vec[1], None]
        lines_z += [origin[2], c_vec[2], None]
        
        lattice_trace = go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color='black', width=5),
            name='Lattice Vectors',
            hoverinfo='none'
        )
        
        # 创建图形
        fig = go.Figure(data=[atom_trace, lattice_trace])
        
        # 更新布局
        fig.update_layout(
            title=f"Crystal Structure: {formula}",
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=700,
            height=600,
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting crystal structure: {str(e)}")
        return None

# 其他函数保持不变...
# [保留之前的 calculate_material_features, filter_selected_features 等函数]

# 如果点击提交按钮
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        # 验证化学公式
        is_valid, validation_msg = validate_chemical_formula(formula_input)
        
        if not is_valid:
            st.error(f"Invalid chemical formula: {validation_msg}")
            st.info("💡 Please use standard chemical notation like: Li7La3Zr2O12, Li10GeP2S12, Li3YCl6")
        else:
            with st.spinner("Processing material and making predictions..."):
                try:
                    # 首先尝试从Materials Project获取晶体结构
                    if mp_api_key and mp_api_key.strip():
                        with st.spinner("Fetching the most relevant crystal structure from Materials Project..."):
                            # 修正化学公式
                            corrected_formula = formula_input.replace('.', '').replace('L1', 'Li').replace('l', 'I')
                            
                            mp_data, mp_error = get_best_materials_project_structure(corrected_formula, mp_api_key)
                            
                            if mp_data and mp_error is None:
                                st.success("✅ Crystal structure retrieved from Materials Project")
                                
                                # 显示材料信息
                                st.subheader("Crystal Structure Information")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Material ID:** {mp_data['material_id']}")
                                    st.write(f"**Formula:** {mp_data['formula']}")
                                    st.write(f"**Space Group:** {mp_data['spacegroup'].get('symbol', 'N/A')} ({mp_data['spacegroup'].get('number', 'N/A')})")
                                    st.write(f"**Structure Type:** {mp_data['structure_type']}")
                                    
                                with col2:
                                    if mp_data['density'] != 'N/A':
                                        st.write(f"**Density:** {mp_data['density']:.2f} g/cm³")
                                    else:
                                        st.write(f"**Density:** N/A")
                                    if mp_data['volume'] != 'N/A':
                                        st.write(f"**Volume:** {mp_data['volume']:.2f} Å³")
                                    else:
                                        st.write(f"**Volume:** N/A")
                                    if mp_data['formation_energy_per_atom'] != 'N/A':
                                        st.write(f"**Formation Energy:** {mp_data['formation_energy_per_atom']:.3f} eV/atom")
                                    else:
                                        st.write(f"**Formation Energy:** N/A")
                                
                                # 绘制晶体结构
                                st.subheader("3D Crystal Structure Visualization")
                                fig = plot_crystal_structure_plotly(mp_data['structure'], corrected_formula)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 显示结构分析
                                    st.info(f"**Structure Analysis:** {mp_data['structure_type'].capitalize()} structure with {mp_data['symmetry']} symmetry")
                            else:
                                st.warning(f"Could not retrieve crystal structure: {mp_error}")
                                st.info("💡 The material might not exist in Materials Project database, or try a different formula")
                    else:
                        st.info("💡 Enter a Materials Project API key to view crystal structure information")
                    
                    # 计算材料特征和预测部分保持不变...
                    # [保留之前的特征计算和预测代码]
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
