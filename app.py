# app.py — 完整版（包含修复 VoronoiNN 初始化问题）
# 功能：formula 查询 Materials Project -> conventional cell -> py3Dmol 渲染
# 包含：MP 官方配色、polyhedral (近似)、supercell、旋转动画、透明度、线框模式
import streamlit as st
import numpy as np
import pandas as pd
import json
import gc
import re
from io import BytesIO

# RDKit / Mordred / AutoGluon (保留你原来的流程)
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from mordred import Calculator, descriptors
from autogluon.tabular import TabularPredictor

# 结构相关
import py3Dmol
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure, Element
# VoronoiNN may vary across pymatgen versions; import it if available
try:
    from pymatgen.analysis.local_env import VoronoiNN
except Exception:
    VoronoiNN = None

# ========== Materials Project API KEY ==========
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"  # 请改为你的 Key（不要公开）

# ========== UI 配置 ==========
st.set_page_config(page_title="Solid Electrolyte Viewer + Predictor", layout="wide")
st.markdown("""
    <style>
    .stApp { max-width: 1100px; margin: 20px auto; padding: 16px; }
    .legend-row { display:flex; gap:12px; flex-wrap:wrap; justify-content:flex-end; margin-top:6px; }
    .legend-item { display:flex; gap:8px; align-items:center; font-size:14px; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Solid Electrolyte Viewer & Ionic Conductivity Predictor")

# ================= Input =================
formula_input = st.text_input("Enter Chemical Formula (formula query to Materials Project)", "Li7La3Zr2O12")
temperature = st.number_input("Temperature (K)", min_value=200, max_value=1000, value=298, step=1)

# visualization options (UI controls)
col1, col2, col3 = st.columns(3)
with col1:
    show_supercell = st.checkbox("Show 2×2×2 supercell", value=False)
    show_polyhedra = st.checkbox("Show polyhedra (approx)", value=False)
with col2:
    enable_spin = st.checkbox("Enable rotation animation", value=False)
    wireframe_mode = st.checkbox("Wireframe mode (thin bonds & smaller spheres)", value=False)
with col3:
    opacity = st.slider("Atom opacity (0-1)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    stick_opacity = st.slider("Bond opacity (0-1)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

submit_button = st.button("Submit and Predict")

# ========== AutoGluon model cache ==========
@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        return TabularPredictor.load("./ag-20251024_075719")
    except Exception as e:
        st.warning(f"Cannot load predictor: {e}")
        return None

# ========== Materials Project 官方色表（接近 MP 风格） ==========
MP_COLOR_MAP = {
    "H": "#FFFFFF","Li":"#CC80FF","Be":"#C2FF00","B":"#FFB5B5","C":"#909090","N":"#3050F8","O":"#FF0D0D",
    "F":"#90E050","Na":"#AB5CF2","Mg":"#8AFF00","Al":"#BFA6A6","Si":"#F0C8A0","P":"#FF8000","S":"#FFFF30",
    "Cl":"#1FF01F","K":"#8F40D4","Ca":"#AFFF8F","Sc":"#BFC2C7","Ti":"#BFC2C7","V":"#A6A6AB","Cr":"#8A99C7",
    "Mn":"#9C7AC7","Fe":"#E06633","Co":"#F090A0","Ni":"#50D050","Cu":"#C88033","Zn":"#7D80B0","Ga":"#C28F8F",
    "Ge":"#668C8C","As":"#BD80E3","Se":"#FFA100","Br":"#A62929","Kr":"#5CB8D1","Rb":"#702EB0","Sr":"#00FF00",
    "Y":"#94FFFF","Zr":"#94E0E0","Nb":"#73C2C9","Mo":"#54B5B5","Tc":"#3B9E9E","Ru":"#248F8F","Rh":"#0A7B7B",
    "Pd":"#006985","Ag":"#C0C0C0","Cd":"#FFD98F","In":"#A67573","Sn":"#668080","Sb":"#9E63A6","Te":"#D47A00",
    "I":"#940094","Xe":"#429EB0","Cs":"#57178F","Ba":"#00C1FF","La":"#70D4FF","Ce":"#FFFFC7","Pr":"#D9FFC7",
    "Nd":"#C7FFC7","Pm":"#A3FFC7","Sm":"#8FFFC7","Eu":"#61FFC7","Gd":"#45FFC7","Tb":"#30FFC7","Dy":"#1FFFC7",
    "Ho":"#00FF9C","Er":"#00E675","Tm":"#00D452","Yb":"#00BF38","Lu":"#00AB24","Hf":"#4DC2FF","Ta":"#4DA6FF",
    "W":"#2194D6","Re":"#267DAB","Os":"#266696","Ir":"#175487","Pt":"#D0D0E0","Au":"#FFD123","Hg":"#B8B8D0",
    "Tl":"#A6544D","Pb":"#575961","Bi":"#9E4FB5","Po":"#AB5C00","At":"#754F45","Rn":"#428296","Fr":"#420066",
    "Ra":"#007D00","Ac":"#70ABFA","Th":"#00BAFF","Pa":"#00A1FF","U":"#008FFF","Np":"#0080FF","Pu":"#006B9A"
}
def mp_color(element):
    return MP_COLOR_MAP.get(element, "#BDBDBD")

# ========== helper: build simplified structure dict for JS ==========
def structure_to_minimal_dict(struct: Structure):
    sites = []
    for site in struct.sites:
        sites.append({"xyz": [float(x) for x in site.coords], "element": str(site.specie)})
    lattice = [[float(x) for x in row] for row in struct.lattice.matrix]
    return {"sites": sites, "lattice": lattice}

# ========== helper: detect bonds by covalent radii ==========
def detect_bonds(struct: Structure, scale_factor=1.2):
    bonds = []
    n = len(struct.sites)
    radii = []
    for s in struct.sites:
        try:
            r = Element(str(s.specie)).covalent_radius
            if r is None or np.isnan(r):
                r = 0.75
        except Exception:
            r = 0.75
        radii.append(r)

    for i in range(n):
        for j in range(i+1, n):
            d = struct.get_distance(i, j)
            cutoff = (radii[i] + radii[j]) * scale_factor
            if d <= cutoff and d > 0.2:
                bonds.append((i, j))
    return bonds

# ========== helper: find coordination (VoronoiNN or fallback) ==========
def get_coordination_list(struct: Structure, max_nn=12):
    """
    For each site, return list of neighbor site indices (VoronoiNN if available, else distance-based fallback).
    """
    # Try VoronoiNN if available
    if VoronoiNN is not None:
        try:
            vn = VoronoiNN()  # default constructor; some pymatgen versions don't accept extra args
            coord_lists = []
            for i in range(len(struct.sites)):
                try:
                    nn_info = vn.get_nn_info(struct, i)
                    neighbors = [int(d['site_index']) for d in nn_info if 'site_index' in d]
                    coord_lists.append(neighbors[:max_nn])
                except Exception:
                    # fallback for this site
                    neighbors = [j for j in range(len(struct.sites)) if j!=i and struct.get_distance(i,j) < 3.0]
                    coord_lists.append(neighbors[:max_nn])
            return coord_lists
        except Exception:
            # If VoronoiNN instantiation fails for some reason, fall back
            pass

    # Fallback: simple distance-based neighbors
    coord_lists = []
    for i in range(len(struct.sites)):
        nbrs = []
        for j in range(len(struct.sites)):
            if i == j:
                continue
            d = struct.get_distance(i, j)
            if d < 3.0:
                nbrs.append(j)
        coord_lists.append(nbrs[:max_nn])
    return coord_lists

# ========== main render function ==========
def render_py3dmol(struct: Structure, *,
                   show_super=False,
                   polyhedra=False,
                   spin=False,
                   atom_opacity=1.0,
                   bond_opacity=1.0,
                   wireframe=False,
                   width=900, height=600):

    if show_super:
        s = struct.copy()
        s.make_supercell([2,2,2])
    else:
        s = struct

    bonds = detect_bonds(s, scale_factor=1.15)
    coord = get_coordination_list(s)

    minimal = structure_to_minimal_dict(s)
    minimal['bonds'] = bonds
    minimal['coord'] = coord

    elements = sorted({site['element'] for site in minimal['sites']})
    color_map = {el: mp_color(el) for el in elements}

    struct_json = json.dumps(minimal)
    color_json = json.dumps(color_map)

    js = f"""
    <div id="viewer" style="width:100%; height:{height}px; position:relative;"></div>
    <script>
    (function() {{
        let data = {struct_json};
        let color_map = {color_json};
        let viewer = $3Dmol.createViewer('viewer', {{ backgroundColor: 'white' }});
        viewer.clear();

        let lattice = data.lattice;
        viewer.addUnitCell({{a: lattice[0], b: lattice[1], c: lattice[2], color:'black', linewidth:1.0}});

        // add atoms
        for (let i=0;i<data.sites.length;i++) {{
            let s = data.sites[i];
            let el = s.element;
            let pos = {{x: s.xyz[0], y: s.xyz[1], z: s.xyz[2]}};
            let color = color_map[el] || "#BBBBBB";
            if ({'true' if wireframe else 'false'}) {{
                viewer.addSphere({{center: pos, radius: 0.18, color: color, opacity: {atom_opacity}}});
            }} else {{
                viewer.addSphere({{center: pos, radius: 0.45, color: color, opacity: {atom_opacity}}});
            }}
        }}

        // add bonds
        for (let b=0;b<data.bonds.length;b++) {{
            let i = data.bonds[b][0], j = data.bonds[b][1];
            let p1 = data.sites[i].xyz, p2 = data.sites[j].xyz;
            let el1 = data.sites[i].element;
            let c = color_map[el1] || "#BBBBBB";
            viewer.addCylinder({{start: {{x:p1[0],y:p1[1],z:p1[2]}}, end: {{x:p2[0],y:p2[1],z:p2[2]}} , radius: {'0.08' if wireframe else '0.15'}, color: c, opacity: {bond_opacity} }});
        }}

        // polyhedra (approx): translucent small spheres at neighbor positions + small cylinders
        if ({'true' if polyhedra else 'false'}) {{
            for (let i=0;i<data.coord.length;i++) {{
                let neighs = data.coord[i];
                let central = data.sites[i].element;
                if (["O","S","Cl","P"].indexOf(central) >= 0) continue;
                for (let idx=0; idx<neighs.length; idx++) {{
                    let j = neighs[idx];
                    let p1 = data.sites[i].xyz, p2 = data.sites[j].xyz;
                    let elj = data.sites[j].element;
                    let c = color_map[elj] || "#BBBBBB";
                    viewer.addSphere({{center: {{x:p2[0],y:p2[1],z:p2[2]}}, radius:0.22, color: c, opacity: 0.18}});
                    viewer.addCylinder({{start: {{x:p1[0],y:p1[1],z:p1[2]}}, end:{{x:p2[0],y:p2[1],z:p2[2]}} , radius:0.06, color: c, opacity: 0.22}});
                }}
            }}
        }}

        if ({'true' if spin else 'false'}) {{
            try {{ viewer.spin(true); }} catch(e){{ console.warn(e); }}
        }}

        viewer.zoomTo();
        viewer.render();
    }})();
    </script>
    """

    st.components.v1.html(js, height=height+40, scrolling=False)

    legend_html = '<div class="legend-row">'
    for el, c in color_map.items():
        legend_html += f'<div class="legend-item"><span style="width:14px;height:14px;border-radius:50%;background:{c};display:inline-block;border:1px solid #333;"></span><span>{el}</span></div>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

# ========== feature calculation (keeps your original approach) ==========
def calculate_material_features(formula):
    try:
        from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry, IonProperty
        from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

        df = pd.DataFrame({"Formula":[formula]})
        df = StrToComposition().featurize_dataframe(df, "Formula", ignore_errors=True)
        if "composition" not in df.columns:
            return {"Formula": formula}
        features = {"Formula": formula}
        df = ElementProperty.from_preset("magpie").featurize_dataframe(df, "composition", ignore_errors=True)
        df = Meredig().featurize_dataframe(df, "composition", ignore_errors=True)
        df = Stoichiometry().featurize_dataframe(df, "composition", ignore_errors=True)
        df = CompositionToOxidComposition().featurize_dataframe(df, "composition", ignore_errors=True)
        df = IonProperty().featurize_dataframe(df, "composition_oxid", ignore_errors=True)
        for col in df.select_dtypes(include=[np.number]).columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0
        return features
    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        return {"Formula": formula}

required_descriptors = [
    'MagpieData mean CovalentRadius',
    'Temp',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence'
]

# ========== helper: get structure from MP (try conventional cell) ==========
def get_structure_by_formula_conventional(formula):
    try:
        with MPRester(MP_API_KEY) as mpr:
            try:
                if hasattr(mpr, "summary") and hasattr(mpr.summary, "search"):
                    res = mpr.summary.search(formula=formula)
                    if res and len(res)>0:
                        first = res[0]
                        if hasattr(first, "structure") and first.structure is not None:
                            struct = first.structure
                        else:
                            matid = None
                            if isinstance(first, dict) and "material_id" in first:
                                matid = first["material_id"]
                            elif hasattr(first, "material_id"):
                                matid = first.material_id
                            if matid:
                                struct = mpr.get_structure_by_material_id(matid)
                        try:
                            struct = struct.get_conventional_standard_structure()
                        except Exception:
                            pass
                        return struct
            except Exception:
                pass
            try:
                q = mpr.query(criteria={"formula": formula}, properties=["material_id"])
                if q:
                    mid = q[0].get("material_id")
                    struct = mpr.get_structure_by_material_id(mid)
                    try:
                        struct = struct.get_conventional_standard_structure()
                    except Exception:
                        pass
                    return struct
            except Exception:
                pass
            try:
                structs = mpr.get_structures(formula)
                if structs:
                    struct = structs[0]
                    try:
                        struct = struct.get_conventional_standard_structure()
                    except Exception:
                        pass
                    return struct
            except Exception:
                pass
    except Exception as e:
        st.error(f"Failed to retrieve structure: {e}")
    return None

# ================== Main UI action ==================
if submit_button:
    if not formula_input:
        st.error("Please enter a chemical formula.")
    else:
        with st.spinner("Fetching structure and computing features..."):
            struct = get_structure_by_formula_conventional(formula_input)
            if struct is None:
                st.warning("No structure found for this formula in Materials Project.")
            else:
                st.subheader("Crystal Structure (from Materials Project)")
                render_py3dmol(struct,
                               show_super=show_supercell,
                               polyhedra=show_polyhedra,
                               spin=enable_spin,
                               atom_opacity=opacity,
                               bond_opacity=stick_opacity,
                               wireframe=wireframe_mode,
                               width=900,
                               height=600)
            features = calculate_material_features(formula_input)
            st.success(f"Total features extracted: {len(features)}")
            sel = {"Temp": float(temperature)}
            for fdesc in required_descriptors:
                if fdesc == "Temp":
                    continue
                sel[fdesc] = features.get(fdesc, 0.0)
            st.subheader("Selected features")
            st.dataframe(pd.DataFrame([sel]))
            predictor = load_predictor()
            if predictor is None:
                st.error("Predictor unavailable.")
            else:
                input_df = pd.DataFrame([{"Formula": formula_input, **sel}])
                models = ["CatBoost","ExtraTreesMSE","LightGBM","KNeighborsDist","WeightedEnsemble_L2","XGBoost"]
                results = {}
                for m in models:
                    try:
                        results[m] = predictor.predict(input_df, model=m)
                    except Exception as e:
                        results[m] = f"Error: {e}"
                st.subheader("Prediction Results")
                st.dataframe(pd.DataFrame(results).iloc[:1,:])
            gc.collect()
