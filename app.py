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
import requests
from PIL import Image
import base64

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
    .structure-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        background-color: white;
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
                          value="Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN")

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

def get_materials_project_structure_with_visualization(formula, api_key):
    """获取Materials Project的晶体结构和可视化数据"""
    if not api_key or not api_key.strip():
        return None, "No API key provided"
    
    try:
        api_key = api_key.strip()
        
        if len(api_key) != 32 or not all(c.isalnum() for c in api_key):
            return None, "Invalid API key format. API key should be 32 alphanumeric characters."
        
        with MPRester(api_key) as mpr:
            # 搜索材料
            entries = mpr.get_entries(formula, inc_structure=True)
            
            if not entries:
                return None, f"No materials found for formula: {formula}"
            
            # 选择第一个材料（通常是最稳定的）
            material = entries[0]
            structure = material.structure
            material_id = material.entry_id
            
            # 获取材料的详细信息
            try:
                summary_results = mpr.summary.search(material_id=material_id, fields=[
                    "formula_pretty", "spacegroup", "density", "volume", 
                    "formation_energy_per_atom", "band_gap", "material_id",
                    "cif", "symmetry"
                ])
                
                if summary_results:
                    material_data = summary_results[0]
                    pretty_formula = material_data.formula_pretty
                    spacegroup_data = material_data.spacegroup
                    spacegroup_symbol = spacegroup_data.symbol if spacegroup_data else "N/A"
                    spacegroup_number = spacegroup_data.number if spacegroup_data else "N/A"
                    density = material_data.density
                    volume = material_data.volume
                    formation_energy = material_data.formation_energy_per_atom
                    band_gap = material_data.band_gap
                    
                    # 获取CIF文件用于可视化
                    cif_data = material_data.cif if hasattr(material_data, 'cif') else None
                    
                else:
                    # 如果summary.search失败，使用基本数据
                    pretty_formula = formula
                    spacegroup_symbol = "N/A"
                    spacegroup_number = "N/A"
                    density = structure.density
                    volume = structure.volume
                    formation_energy = material.energy_per_atom
                    band_gap = "N/A"
                    cif_data = None
                    
            except Exception as detail_error:
                # 如果获取详细信息失败，使用基本数据
                pretty_formula = formula
                spacegroup_symbol = "N/A"
                spacegroup_number = "N/A"
                density = structure.density
                volume = structure.volume
                formation_energy = material.energy_per_atom
                band_gap = "N/A"
                cif_data = None
            
            return {
                'structure': structure,
                'material_id': material_id,
                'spacegroup': {
                    'symbol': spacegroup_symbol,
                    'number': spacegroup_number
                },
                'density': density,
                'volume': volume,
                'formation_energy_per_atom': formation_energy,
                'band_gap': band_gap,
                'formula': formula,
                'pretty_formula': pretty_formula,
                'cif_data': cif_data
            }, None
            
    except Exception as e:
        return None, f"Error accessing Materials Project: {str(e)}"

def get_materials_project_visualization_url(material_id, api_key):
    """获取Materials Project官方可视化URL"""
    try:
        # Materials Project的官方可视化URL格式
        base_url = "https://next-gen.materialsproject.org"
        visualization_url = f"{base_url}/materials/{material_id}"
        
        return visualization_url
    except Exception as e:
        return None

def display_materials_project_visualization(material_id, api_key):
    """显示Materials Project的官方晶体结构可视化"""
    try:
        # 获取可视化URL
        viz_url = get_materials_project_visualization_url(material_id, api_key)
        
        if viz_url:
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <a href="{viz_url}" target="_blank" style="
                    display: inline-block;
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: bold;
                ">
                🎯 View Interactive Crystal Structure on Materials Project
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            # 显示嵌入的iframe（可选）
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <small>Click the button above to view the interactive crystal structure on Materials Project website</small>
            </div>
            """, unsafe_allow_html=True)
            
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"Error displaying Materials Project visualization: {str(e)}")
        return False

def create_enhanced_structure_plot(structure, formula, material_id):
    """创建增强的晶体结构3D图 - 使用pymatgen的VESTA风格可视化"""
    try:
        from pymatgen.vis.structure_vtk import StructureVis
        import tempfile
        import os
        
        # 创建临时文件保存结构图像
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # 使用pymatgen的可视化功能
        vis = StructureVis(show_polyhedron=False)
        vis.set_structure(structure)
        vis.zoom_to_fit()
        
        # 保存图像（这里简化处理，实际可能需要更复杂的渲染）
        # 由于streamlit中直接使用pymatgen可视化比较复杂，我们回退到plotly方法
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            
        # 使用改进的plotly可视化
        return create_simple_structure_plot(structure, formula, material_id)
        
    except Exception as e:
        st.warning(f"Enhanced visualization failed, using basic method: {str(e)}")
        return create_simple_structure_plot(structure, formula, material_id)

def create_simple_structure_plot(structure, formula, material_id):
    """创建简化的晶体结构3D图"""
    try:
        # 获取晶格参数
        lattice = structure.lattice
        sites = structure.sites
        
        # 创建原子位置数据
        x, y, z = [], [], []
        colors, sizes, symbols, hover_texts = [], [], [], []
        
        # 改进的原子颜色映射（VESTA风格）
        color_map = {
            'Li': '#CC80FF', 'La': '#70D4FF', 'Zr': '#4EACCE', 'O': '#FF0D0D',
            'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F', 'Ge': '#668F8F',
            'Y': '#94FFFF', 'F': '#90E050', 'Br': '#A62929', 'I': '#940094',
            'Na': '#AB5CF2', 'K': '#8F40D4', 'Mg': '#8AFF00', 'Ca': '#3DFF00',
            'Al': '#BFA6A6', 'Si': '#F0C8A0', 'Ti': '#BFC2C7', 'Fe': '#E06633',
            'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'B': '#F0B0B0'
        }
        
        # 原子大小映射
        size_map = {
            'Li': 8, 'La': 15, 'Zr': 12, 'O': 10,
            'P': 11, 'S': 10, 'Cl': 10, 'Ge': 12,
            'Y': 12, 'F': 8, 'Br': 11, 'I': 13,
            'Na': 10, 'K': 12, 'Mg': 11, 'Ca': 12,
            'Al': 11, 'Si': 11, 'Ti': 12, 'Fe': 12,
            'H': 6, 'C': 10, 'N': 9, 'B': 9
        }
        
        for i, site in enumerate(sites):
            x.append(site.coords[0])
            y.append(site.coords[1])
            z.append(site.coords[2])
            element = site.species_string
            colors.append(color_map.get(element, '#CCCCCC'))
            sizes.append(size_map.get(element, 10))
            symbols.append(element)
            hover_texts.append(f"{element} atom<br>Position: ({site.coords[0]:.2f}, {site.coords[1]:.2f}, {site.coords[2]:.2f})")
        
        # 创建原子轨迹
        atom_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.95,
                line=dict(width=2, color='darkgray')
            ),
            text=symbols,
            textposition="middle center",
            textfont=dict(size=10, color='black', family="Arial", weight="bold"),
            hoverinfo='text',
            hovertext=hover_texts,
            name='Atoms'
        )
        
        # 创建晶格线
        lattice_traces = []
        
        # 晶格向量
        origin = [0, 0, 0]
        a_vec = lattice.matrix[0]
        b_vec = lattice.matrix[1]
        c_vec = lattice.matrix[2]
        
        # a轴 - 红色
        lattice_traces.append(go.Scatter3d(
            x=[origin[0], a_vec[0]],
            y=[origin[1], a_vec[1]],
            z=[origin[2], a_vec[2]],
            mode='lines',
            line=dict(color='red', width=8),
            name='a-axis',
            hoverinfo='none',
            showlegend=False
        ))
        
        # b轴 - 绿色
        lattice_traces.append(go.Scatter3d(
            x=[origin[0], b_vec[0]],
            y=[origin[1], b_vec[1]],
            z=[origin[2], b_vec[2]],
            mode='lines',
            line=dict(color='green', width=8),
            name='b-axis',
            hoverinfo='none',
            showlegend=False
        ))
        
        # c轴 - 蓝色
        lattice_traces.append(go.Scatter3d(
            x=[origin[0], c_vec[0]],
            y=[origin[1], c_vec[1]],
            z=[origin[2], c_vec[2]],
            mode='lines',
            line=dict(color='blue', width=8),
            name='c-axis',
            hoverinfo='none',
            showlegend=False
        ))
        
        # 创建图形
        all_traces = [atom_trace] + lattice_traces
        
        fig = go.Figure(data=all_traces)
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"Crystal Structure: {formula}",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor='white'
            ),
            width=700,
            height=600,
            margin=dict(l=20, r=20, b=20, t=60),
            showlegend=False
        )
        
        # 添加坐标轴样式
        fig.update_scenes(
            xaxis=dict(
                backgroundcolor="white", 
                gridcolor="lightgray", 
                showbackground=True,
                showgrid=True
            ),
            yaxis=dict(
                backgroundcolor="white", 
                gridcolor="lightgray", 
                showbackground=True,
                showgrid=True
            ),
            zaxis=dict(
                backgroundcolor="white", 
                gridcolor="lightgray", 
                showbackground=True,
                showgrid=True
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating structure plot: {str(e)}")
        return None

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
                        with st.spinner("Fetching crystal structure from Materials Project..."):
                            # 修正化学公式
                            corrected_formula = formula_input.replace('.', '').replace('L1', 'Li').replace('l', 'I').replace('3272', '3Zr2')
                            
                            mp_data, mp_error = get_materials_project_structure_with_visualization(corrected_formula, mp_api_key)
                            
                            if mp_data and mp_error is None:
                                st.success("✅ Crystal structure retrieved from Materials Project")
                                
                                # 显示材料信息
                                st.subheader("📊 Crystal Structure Information")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Material ID:** `{mp_data['material_id']}`")
                                    st.write(f"**Formula:** {mp_data['pretty_formula']}")
                                    st.write(f"**Space Group:** {mp_data['spacegroup'].get('symbol', 'N/A')} ({mp_data['spacegroup'].get('number', 'N/A')})")
                                    
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
                                
                                # 分析结构特征
                                structure_info = analyze_structure_features(mp_data['structure'])
                                
                                # 显示结构分析
                                st.subheader("🔬 Structure Analysis")
                                col3, col4 = st.columns(2)
                                with col3:
                                    st.write(f"**Structure Type:** {structure_info['structure_type'].capitalize()}")
                                with col4:
                                    st.write(f"**Symmetry:** {structure_info['symmetry'].capitalize()}")
                                
                                # 显示Materials Project官方可视化链接
                                st.subheader("🎯 Interactive Crystal Structure")
                                
                                # 显示官方可视化链接
                                display_materials_project_visualization(mp_data['material_id'], mp_api_key)
                                
                                # 同时显示本地的3D可视化
                                st.subheader("🔍 3D Structure Preview")
                                
                                # 创建并显示3D结构图
                                fig = create_enhanced_structure_plot(
                                    mp_data['structure'], 
                                    mp_data['pretty_formula'], 
                                    mp_data['material_id']
                                )
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 添加交互说明
                                    st.info("""
                                    **💡 Interactive Controls for 3D Preview:**
                                    - **Rotate:** Click and drag to rotate the structure
                                    - **Zoom:** Use mouse wheel to zoom in/out
                                    - **Pan:** Hold Shift and drag to pan
                                    - **Reset:** Double-click to reset view
                                    - **Hover:** Hover over atoms to see details
                                    
                                    **🎯 For full interactive features:** Click the button above to view on Materials Project
                                    """)
                                else:
                                    st.warning("3D preview not available, please use the Materials Project link above")
                                
                            else:
                                st.warning(f"Could not retrieve crystal structure: {mp_error}")
                                st.info("💡 The material might not exist in Materials Project database, or try a different formula")
                    else:
                        st.info("💡 Enter a Materials Project API key to view crystal structure information")
                    
                    # 其余代码保持不变...
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
