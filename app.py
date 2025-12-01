import streamlit as st
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.vis.structure_3d import Structure3D
import json

# -----------------------------
# Streamlit 页面基本设置
# -----------------------------
st.set_page_config(page_title="Crystal Viewer", layout="wide")
st.title("🔬 Crystal Structure Viewer (Materials Project)")

API_KEY = st.text_input("请输入你的 Materials Project API Key（必填）", type="password")

mp_id = st.text_input("输入材料的 Materials Project ID，例如：mp-149（Si）")

# =============================
# 获取结构函数
# =============================
def fetch_structure(mp_id: str, api_key: str):
    try:
        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(mp_id)
        # 标准化晶胞，让结构更“端正”
        structure = structure.get_primitive_structure()
        structure = structure.get_reduced_structure()
        return structure
    except Exception as e:
        st.error(f"❌ 获取结构失败：{e}")
        return None


# =============================
# 颜色生成：按元素固定颜色
# =============================
def get_element_colors(structure):
    """
    按元素生成颜色，统一用于 3Dmol 和图例
    """
    unique_elements = sorted({str(site.specie) for site in structure})
    cmap = {}
    np.random.seed(42)

    for elem in unique_elements:
        # 生成稳定固定的颜色
        r = np.random.randint(100, 255)
        g = np.random.randint(80, 255)
        b = np.random.randint(80, 255)
        cmap[elem] = f"#{r:02X}{g:02X}{b:02X}"

    return cmap


# =============================
# 3Dmol.js 渲染函数
# =============================
def render_structure(structure, color_map):
    struct_json = structure.to_json()

    script = f"""
        <div id="container" style="width: 100%; height: 500px;"></div>
        <script>
            let structData = {struct_json};

            let viewer = $3Dmol.createViewer('container', {{
                backgroundColor: 'white'
            }});

            let atoms = structData['sites'];

            atoms.forEach(site => {{
                let pos = site['abc'];
                let elem = site['label'];

                viewer.addSphere({{
                    center: {{
                        x: pos[0],
                        y: pos[1],
                        z: pos[2]
                    }},
                    radius: 0.4,
                    color: "{color_map.get(elem, "#FF0000")}"
                }});
            }});

            viewer.zoomTo();
            viewer.render();
        </script>
    """

    st.components.v1.html(script, height=520)


# =============================
# 图例 HTML 生成
# =============================
def create_legend(color_map):
    html = """
    <div style="display:flex; flex-direction:column; align-items:flex-start; margin-top:10px;">
    """

    for elem, color in color_map.items():
        html += f"""
        <div style="display:flex; align-items:center; margin:3px;">
            <span style="width:16px; height:16px; background:{color}; 
                         display:inline-block; border-radius:50%; margin-right:8px;"></span>
            <span style="font-size:16px;">{elem}</span>
        </div>
        """

    html += "</div>"
    return html


# =============================
# 主按钮执行
# =============================
if st.button("🔍 获取并显示晶体结构"):
    if not API_KEY or not mp_id:
        st.warning("⚠ 请填入 API Key 和 MP ID")
    else:
        structure = fetch_structure(mp_id, API_KEY)
        if structure:
            st.subheader("📦 Crystal Structure (3D Viewer)")
            
            color_map = get_element_colors(structure)

            render_structure(structure, color_map)

            st.subheader("🎨 元素颜色图例")
            st.markdown(create_legend(color_map), unsafe_allow_html=True)
