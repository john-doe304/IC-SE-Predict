# app.py — Enhanced crystal visualization + features + predictions

import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DSVG
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile
import base64
from io import BytesIO
import gc
import re
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import json
import math
import warnings
warnings.filterwarnings("ignore")

# visualization
import py3Dmol
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

# ===== Materials Project API key (replace with your own) =====
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"

# ============ UI styling ============
st.set_page_config(layout="centered", page_title="Solid Electrolyte Predictor")
st.markdown(
    """
    <style>
    .stApp { max-width: 1100px; margin: 20px auto; }
    .rounded { border: 2px solid #ddd; border-radius:10px; padding:12px; margin-bottom:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predict Ionic Conductivity & Advanced Crystal Viewer")
st.markdown("Enter a chemical formula (or upload a CIF) and explore structure, polyhedra, supercells, bands/DOS and prediction.")

# ---------- Inputs ----------
left, right = st.columns([2,1])
with left:
    formula_input = st.text_input("Enter Chemical Formula (or leave blank if uploading CIF):", placeholder="e.g., Li7La3Zr2O12")
    uploaded_cif = st.file_uploader("Upload CIF file (optional)", type=["cif"])
    upload_charge = st.file_uploader("Upload volumetric file (cube/CHGCAR) for isosurface (optional)", type=["cube","CHGCAR","vasprun.xml"])
    upload_orbit = st.file_uploader("Upload orbital cube file (optional)", type=["cube"])
    upload_procar = st.file_uploader("Upload PROCAR (orbital projection) (optional)", type=["PROCAR","procar","txt"])
    temp = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=10)
    submit = st.button("Submit and Predict")

with right:
    st.subheader("View Options")
    cell_type = st.selectbox("Cell type to display", ["conventional (MP style)", "primitive", "primitive + supercell"], index=0)
    supercell_multiplier = st.slider("Supercell multiplier (when supercell selected)", 1, 4, 2)
    show_poly = st.checkbox("Highlight coordination polyhedra (CrystalNN)", value=True)
    show_poly_only = st.checkbox("Show polyhedra only (hide sticks)", value=False)
    auto_rotate = st.checkbox("Auto-rotate animation", value=True)
    highlight_elem = st.selectbox("Highlight element (or All)", options=["All","Li","La","Zr","O","Cl","Y"], index=0)
    color_scheme = st.selectbox("Color scheme", ["default", "mp-style (green/red)"], index=0)

# required descriptors
required_descriptors = ['MagpieData mean CovalentRadius', 'Temp', 'MagpieData avg_dev SpaceGroupNumber',
                        '0-norm', 'MagpieData mean MeltingT', 'MagpieData avg_dev Column', 'MagpieData mean NValence']

@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        return TabularPredictor.load("./ag-20251024_075719")
    except Exception:
        return None

# ---------- Helper: get structure (robust using new MP API) ----------
def fetch_structure_from_mp(formula):
    if not formula:
        return None, None
    try:
        with MPRester(MP_API_KEY) as mpr:
            # summary search (new API)
            try:
                results = mpr.materials.summary.search(formula=formula)
                if not results:
                    return None, None
                mid = results[0].material_id
                # get structure dict
                sdict = mpr.materials.structures.get(mid)
                if "structure" not in sdict:
                    return None, None
                struct = Structure.from_dict(sdict["structure"])
                return struct, mid
            except Exception as e:
                st.warning(f"MP query failed: {e}")
                return None, None
    except Exception as e:
        st.error(f"Failed to contact Materials Project: {e}")
        return None, None

# ---------- Helper: read uploaded CIF ----------
def read_cif(uploaded):
    try:
        txt = uploaded.read().decode()
        s = Structure.from_str(txt, fmt="cif")
        return s
    except Exception as e:
        st.error(f"Failed to read CIF: {e}")
        return None

# ---------- Helper: convert to desired cell ----------
def prepare_structure(struct, cell_choice, multiplier=1):
    s = struct.copy()
    try:
        if cell_choice == "conventional (MP style)":
            try:
                s = s.get_conventional_standard_structure()
            except Exception:
                pass
        elif cell_choice == "primitive":
            try:
                s = s.get_primitive_structure()
            except Exception:
                pass
        # supercell option
        if multiplier > 1:
            s.make_supercell([multiplier, multiplier, multiplier])
        return s
    except Exception as e:
        st.warning(f"Structure prepare error: {e}")
        return struct

# ---------- Helper: create py3Dmol HTML with optional JS animation and polyhedra rendering ----------
def render_py3dmol_from_structure(structure, auto_rotate=True, poly_info=None, highlight_elem="All", color_scheme="default", show_poly_only=False):
    """
    structure: pymatgen Structure
    poly_info: list of dicts {'center_index':i, 'neighbors':[j,...], 'cn':n, 'type': 'oct'/'tet'/...}
    """
    cif = structure.to(fmt="cif")
    view = py3Dmol.view(width=900, height=620)
    view.addModel(cif, "cif")
    # default style
    view.setStyle({'stick':{}})
    # color scheme
    if color_scheme == "mp-style":
        # attempt to color Li small, La green, Zr green, O red-ish
        view.setStyle({'elem':'Li'},{'sphere':{'scale':0.2}, 'stick':{'radius':0.12}})
        view.setStyle({'elem':'La'},{'sphere':{'scale':0.45}, 'stick':{'radius':0.2}, 'color':'green'})
        view.setStyle({'elem':'Zr'},{'sphere':{'scale':0.45}, 'stick':{'radius':0.2}, 'color':'green'})
        view.setStyle({'elem':'O'},{'sphere':{'scale':0.35}, 'stick':{'radius':0.15}, 'color':'red'})
    else:
        # default spheres + sticks
        view.setStyle({'sphere':{'scale':0.3}, 'stick':{'radius':0.12}})
    # highlight element by making it larger/colored
    if highlight_elem != "All":
        view.setStyle({'elem':highlight_elem},{'sphere':{'scale':0.5}, 'stick':{'radius':0.2}, 'color':'yellow'})

    # draw polyhedra: use neighbor coordinates to draw translucent convex hull as triangular faces
    # py3Dmol doesn't provide mesh-from-points directly, so approximate by adding cylinders for edges and enlarged spheres for neighbors
    if poly_info:
        # optionally hide underlying sticks
        if show_poly_only:
            view.setStyle({'stick':{}}, {'stick':{'radius':0.0}})
        for poly in poly_info:
            center_idx = poly['center_index']
            neighs = poly['neighbors']
            # obtain positions from structure
            center_coord = structure[center_idx].coords
            # draw center as highlighted sphere
            view.addSphere({'center':{'x':float(center_coord[0]), 'y':float(center_coord[1]), 'z':float(center_coord[2])},
                            'radius':0.5, 'color':'orange', 'opacity':0.6})
            # draw neighbor spheres and cylinders
            for nidx in neighs:
                nc = structure[nidx].coords
                # sphere
                view.addSphere({'center':{'x':float(nc[0]), 'y':float(nc[1]), 'z':float(nc[2])},
                                'radius':0.35, 'color':'lightgreen', 'opacity':0.5})
                # cylinder between center and neighbor
                view.addCylinder({'start':{'x':float(center_coord[0]), 'y':float(center_coord[1]), 'z':float(center_coord[2])},
                                  'end':{'x':float(nc[0]), 'y':float(nc[1]), 'z':float(nc[2])},
                                  'radius':0.08, 'color':'darkgray', 'opacity':1.0})
    view.addUnitCell()
    view.zoomTo()
    html = view._make_html()
    # Add JS for auto-rotate if requested: use viewer.setInterval to rotate
    if auto_rotate:
        # find the viewer var name inside html (py3Dmol sets variable "viewer" or "gldiv" wrapper). We'll append a small script to rotate.
        rotate_js = """
        <script>
        (function(){
            try{
                // find the first 3Dmol viewer on page
                for (const key in window) {
                    // skip long enumeration
                }
            }catch(e){}
            // simple approach: rotate using requestAnimationFrame
            let found=false;
            const tryAttach = () => {
                const ivs = document.querySelectorAll('[id^="3dmol"]');
                if(ivs.length>0){
                    const div=ivs[0];
                    const canvas = div.querySelector('canvas');
                    if(!canvas) { setTimeout(tryAttach,200); return; }
                    // get the viewer instance (3Dmol stores as div.viewer)
                    const v = div.viewer;
                    if(!v){ setTimeout(tryAttach,200); return; }
                    found=true;
                    let angle=0;
                    function spin(){
                        v.rotate(1, 'y'); // rotate 1 degree around y
                        v.render();
                        requestAnimationFrame(spin);
                    }
                    requestAnimationFrame(spin);
                } else {
                    setTimeout(tryAttach,200);
                }
            };
            tryAttach();
        })();
        </script>
        """
        html += rotate_js
    st.components.v1.html(html, height=660, scrolling=True)

# ---------- compute coordination (CrystalNN) ----------
def compute_coordination(structure, max_sites=50):
    """
    returns list of polyhedra info: {'center_index':i,'neighbors':[j,...],'cn':n,'type':str}
    uses CrystalNN for neighbor finding
    """
    try:
        cnn = CrystalNN()
        poly_list = []
        nsites = min(len(structure), max_sites)
        for i in range(nsites):
            try:
                nn = cnn.get_nn_info(structure, i)
                neigh_indices = [d['site_index'] for d in nn]
                cn = len(neigh_indices)
                # try to guess poly type (oct/tet) by cn
                ptype = 'other'
                if cn == 6:
                    ptype = 'oct'
                elif cn == 4:
                    ptype = 'tet'
                poly_list.append({'center_index':i, 'neighbors':neigh_indices, 'cn':cn, 'type':ptype})
            except Exception:
                continue
        return poly_list
    except Exception as e:
        st.warning(f"Coordination computation failed: {e}")
        return None

# ---------- Try to fetch bandstructure & dos from MP (best-effort) ----------
def fetch_band_and_dos(material_id):
    try:
        with MPRester(MP_API_KEY) as mpr:
            bs = None
            dos = None
            # try bandstructure endpoint
            try:
                bdict = mpr.materials.bandstructure.get(material_id)
                if bdict and 'bandstructure' in bdict:
                    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
                    bs = BandStructureSymmLine.from_dict(bdict['bandstructure'])
            except Exception:
                bs = None
            # try dos
            try:
                dd = mpr.materials.dos.get(material_id)
                if dd and 'dos' in dd:
                    from pymatgen.electronic_structure.dos import CompleteDos
                    dos = CompleteDos.from_dict(dd['dos'])
            except Exception:
                dos = None
            return bs, dos
    except Exception:
        return None, None

# ---------- Simple plotting functions for band and DOS ----------
def plot_dos(dos):
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        energies = np.array([e.value for e in dos.energies])
        total = np.array([dos.get_densities(etype='total')[i] for i in range(len(dos.energies))])
        ax.plot(total, energies, label='Total DOS')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.7)
        ax.set_xlabel("DOS")
        ax.set_ylabel("Energy (eV)")
        ax.invert_yaxis()
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Plot DOS failed: {e}")

def plot_band(bs):
    try:
        from pymatgen.electronic_structure.plotter import BSPlotter
        plotter = BSPlotter(bs)
        fig = plotter.get_plot()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Plot bandstructure failed: {e}")

# ---------- Isosurface from uploaded volumetric file (best-effort) ----------
def try_show_isosurface_from_cube(uploaded_file, structure):
    """
    Try to parse cube/CHGCAR and display isosurface; this is best-effort and may fail for very large files.
    """
    try:
        txt = uploaded_file.read()
        # We'll try to use py3Dmol volumetric API by embedding the raw cube as text inside the 3Dmol viewer.
        # 3Dmol supports addVolumetricData / addIsosurface but py3Dmol python wrapper doesn't expose all; we'll fallback to show link and instructions.
        st.info("Isosurface upload detected. Attempting to show isosurface (best-effort). If it fails, download the file and view in VESTA or Jmol.")
        # Save to temp and provide download link and instruction
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cube")
        tmp.write(txt)
        tmp.close()
        st.success(f"Volumetric file saved to {tmp.name}. You can download it below and view in local tools (VESTA).")
        with open(tmp.name, "rb") as f:
            b = f.read()
            st.download_button("Download uploaded volumetric file", data=b, file_name=os.path.basename(tmp.name))
    except Exception as e:
        st.warning(f"Isosurface display not available in this environment: {e}")

# ---------- Main logic on submit ----------
if submit:
    struct = None
    material_id = None

    # 1) If a CIF uploaded, use it; else try MP
    if uploaded_cif is not None:
        struct = read_cif(uploaded_cif)
        material_id = None
    else:
        struct, material_id = fetch_structure_from_mp(formula_input)

    if struct is None:
        st.error("No structure available (check formula or upload CIF).")
    else:
        # Prepare structure according to user's choice
        mult = supercell_multiplier if cell_type.endswith("supercell") else 1
        s_prepared = prepare_structure(struct, cell_choice=("conventional (MP style)" if cell_type.startswith("conventional") else "primitive"), multiplier=mult)

        # Compute polyhedra info (CrystalNN) if requested
        poly_info = None
        if show_poly:
            with st.spinner("Computing coordination (CrystalNN)..."):
                poly_info = compute_coordination(s_prepared)

        # Render 3D
        st.subheader("Crystal Structure Viewer")
        render_py3dmol_from_structure(s_prepared, auto_rotate=auto_rotate, poly_info=poly_info if show_poly else None, highlight_elem=highlight_elem, color_scheme=color_scheme, show_poly_only=show_poly_only)

        # Provide CIF download
        cif_bytes = s_prepared.to(fmt="cif").encode()
        st.download_button("Download displayed CIF", data=cif_bytes, file_name=f"{formula_input or 'structure'}.cif")

        # If volumetric uploaded, attempt to show isosurface (best-effort)
        if upload_charge is not None:
            try_show_isosurface_from_cube(upload_charge, s_prepared)

        # If orbital cube uploaded, provide download / instructions
        if upload_orbit is not None:
            st.info("Orbital cube uploaded. For orbital isosurface visualization, use VMD/VESTA or Jmol locally. This app can save the file for download.")
            data = upload_orbit.read()
            st.download_button("Download uploaded orbital cube", data=data, file_name="orbital.cube")

        # If PROCAR uploaded, save and offer instructions
        if upload_procar is not None:
            st.info("PROCAR uploaded. You can analyze orbital projections locally (sum projections) or use pymatgen/vasp tools offline.")
            st.download_button("Download PROCAR", data=upload_procar.read(), file_name="PROCAR")

        # Attempt to fetch band/DOS from Materials Project if we have material_id
        if material_id is not None:
            with st.spinner("Attempting to fetch bandstructure & DOS from Materials Project (best-effort)..."):
                try:
                    bs, dos = fetch_band_and_dos(material_id)
                    if bs is None and dos is None:
                        st.info("No bandstructure/DOS available in Materials Project for this material (or MP API didn't return it).")
                    else:
                        if dos is not None:
                            st.subheader("Density of States (DOS)")
                            plot_dos(dos)
                        if bs is not None:
                            st.subheader("Band Structure")
                            plot_band(bs)
                except Exception as e:
                    st.warning(f"Band/DOS retrieval failed: {e}")
        else:
            st.info("No Materials Project material id available (CIF upload). Band/DOS fetch skipped.")

        # ==== Feature extraction & prediction (your existing pipeline) ====
        with st.spinner("Calculating composition features and predicting..."):
            # composition features using your calculate_material_features (re-implement inline to avoid circular)
            try:
                from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry, IonProperty
                from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
                dfc = pd.DataFrame({"Formula":[formula_input]})
                stc = StrToComposition()
                dfc = stc.featurize_dataframe(dfc, "Formula", ignore_errors=True)
                if "composition" in dfc.columns and dfc['composition'].iloc[0] is not None:
                    ep = ElementProperty.from_preset('magpie')
                    dfc = ep.featurize_dataframe(dfc, 'composition', ignore_errors=True)
                    dfc = Meredig().featurize_dataframe(dfc, 'composition', ignore_errors=True)
                    dfc = Stoichiometry().featurize_dataframe(dfc, 'composition', ignore_errors=True)
                    dfc = CompositionToOxidComposition().featurize_dataframe(dfc, 'composition', ignore_errors=True)
                    dfc = IonProperty().featurize_dataframe(dfc, 'composition_oxid', ignore_errors=True)
                    numeric_columns = dfc.select_dtypes(include=[np.number]).columns
                    features = {}
                    for col in numeric_columns:
                        val = dfc[col].iloc[0]
                        features[col] = float(val) if not pd.isna(val) else 0.0
                else:
                    features = {"Formula": formula_input}
                st.write(f"✅ Total features extracted: {len(features)}")
                # show selected descriptors
                selected = {k:(features.get(k, 0.0) if k!='Temp' else float(temp)) for k in required_descriptors}
                selected['Temp'] = float(temp)
                st.subheader("Selected Features")
                st.dataframe(pd.DataFrame([selected]))
                # Prepare input df
                input_df = pd.DataFrame([{**{k:features.get(k,0.0) for k in features}, "Formula":formula_input, "Temp":float(temp)}])
            except Exception as e:
                st.warning(f"Feature extraction failed: {e}")
                input_df = pd.DataFrame({"Formula":[formula_input], "Temp":[float(temp)]})

            # predict with autogluon
            predictor = load_predictor()
            if predictor is None:
                st.error("AutoGluon predictor not available on this server (check model path).")
            else:
                models = ['CatBoost','ExtraTreesMSE','LightGBM','KNeighborsDist','WeightedEnsemble_L2','XGBoost']
                preds = {}
                for m in models:
                    try:
                        preds[m] = predictor.predict(input_df, model=m)
                    except Exception as e:
                        preds[m] = f"Error: {e}"
                st.subheader("Prediction Results")
                st.dataframe(pd.DataFrame(preds).iloc[:1,:])

    # end of submit processing

# ---------- Footer & notes ----------
st.markdown("---")
st.markdown("**Notes & Limitations:**")
st.markdown("""
- Charge density isosurfaces / orbital visualizations require precomputed volumetric data (cube/CHGCAR) or DFT outputs — this app supports uploading these files and will provide downloads or best-effort display, but heavy volumetric rendering is better done in local tools (VESTA, VMD).  
- Band structure / DOS retrieval is attempted from the Materials Project API; many materials do not have bandstructure/DOS stored. For full band/DOS plotting you may need local DFT outputs (vasprun.xml) or MP-provided data.  
- For precise polyhedra (polyhedron faces), this app approximates polyhedra by highlighted atoms and bonds; constructing exact triangular face meshes requires advanced meshing (outside 3Dmol simple wrapper) and can be added if you provide volumetric/polyhedral files.  
- If deployment environment cannot install heavy packages (rdkit, autogluon), use local conda environment with conda-forge to install them.  
""")

