import streamlit as st
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure

# -------------------------------------------
# Streamlit 基本设置
# -------------------------------------------
st.set_page_config(page_title="Crystal Structure Viewer", layout="wide")
st.title("🔬 Crystal Structure Viewer (Materials Project)")

# 输入 API Key 和 MP 材料 ID
api_key = st.text_input("请输入 Materials Project API Key（必填）", type="password")
mp_id = st.text_input("请输入 Materials Project ID，例如：mp-942733")


# ======================================================
# 函数：从 Materials Project 获取结构
# ======================================================
def fetch_structure(mp_id, api_key):
    try:
        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(mp_id)

        # 标准化晶体结构，让渲染更接近 MP
        structure = structure.get_reduced_structure()
        structure = structure.get_primitive_structure()

        return structure

    except Exception as e:
        st.error(f"❌ 结构获取失败：{e}")
        return None


# ======================================================
# 颜色：按元素固定颜色（稳定渲染）
# ======================================================
def get_element_colors(structure):
    elements = sorted({str(site.specie) for site in structure})
    color_map = {}

    # 固定随机种子 → 每次颜色一致
    np.random.seed(2025)

    for elem in elements:
        r = np.random.randint(60, 230)
        g = np.random.randint(60, 230)
        b = np.random.randint(60, 230)
        color_map[elem] = f"#{r:02X}{g:02X}{b:02X}"

    return color_map


# ======================================================
# 渲染晶体结构（使用 3Dmol.js）
# ======================================================
def render_structure(structure, color_map):
    struct_json = structure.to_json()

    html = f"""
        <div id="viewer" style="width: 100%; height: 500px;"></div>

        <script>
            // 初始化 3Dmol Viewer
            var viewer = $3Dmol.createViewer("viewer", {{
                backgroundColor: "white"
            }});

            let struct = {struct_json};

            // 添加原子球体
            struct.sites.forEach(site => {{
                let pos = site.xyz;
                let elem = site.label;

                viewer.addSphere({{
                    center: {{x: pos[0], y: pos[1], z: pos[2]}},
                    radius: 0.45,
                    color: "{color_map.get(elem, "#FF0000")}"
                }});
            }});

            // 晶胞框
            let matrix = struct.lattice.matrix;
            viewer.addUnitCell({{
                a: matrix[0],
                b: matrix[1],
                c: matrix[2],
                color: "black",
                linewidth: 1.2
            }});

            viewer.zoomTo();
            viewer.render();
        </script>
    """

    st.components.v1.html(html, height=500)


# ======================================================
# 图例生成（类似 Materials Project 右下角）
# ======================================================
def render_legend(color_map):
    legend_html = """
        <div style="position: relative; margin-top: 15px;">
    """

    for elem, color in color_map.items():
        legend_html += f"""
            <div style="display: flex; align-items: center; margin: 6px;">
                <div style="
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background: {color};
                    margin-right: 8px;
                "></div>
                <span style="font-size: 15px;">{elem}</span>
            </div>
        """

    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


# ======================================================
# 主程序按钮
# ======================================================
if st.button("📦 显示晶体结构"):
    if not api_key or not mp_id:
        st.warning("⚠ 请先输入 API Key 和材料 ID")
    else:
        structure = fetch_structure(mp_id, api_key)

        if structure:
            st.subheader("📡 Crystal Structure from Materials Project")
            color_map = get_element_colors(structure)

            render_structure(structure, color_map)

            st.subheader("🎨 元素颜色图例")
            render_legend(color_map)
