# ===== app.py: MP-like crystal viewer + auto-bond + features + prediction =====
# Compatibility shim: restore numpy.product if missing
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

# RDKit (kept from your original)
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DSVG

# Mordred / Matminer (kept)
from mordred import Calculator, descriptors

# Pymatgen
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

# For automatic neighbor finding (bonds)
from pymatgen.analysis.local_env import CrystalNN

# Optional convex hull / scipy usage (polyhedron, not required)
try:
    from scipy.spatial import ConvexHull
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# === Materials Project API Key (embedded per your request) ===
MP_API_KEY = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"

# === Streamlit page style ===
st.set_page_config(page_title="MP-like Crystal Viewer", layout="centered")
st.markdown("""
    <style>
    .stApp { border: 2px solid #808080; border-radius: 12px; margin: 30px auto; max-width: 1100px; padding: 18px; }
    </style>
""", unsafe_allow_html=True)

st.title("Predict Ionic Conductivity — MP-like Structure Viewer")
st.caption("Fetch structure from Materials Project (or COD), visualize MP-like (auto bonds), extract features and predict.")

# === Inputs ===
formula_input = st.text_input("Chemical formula (or MP id e.g. mp-1234):", placeholder="Li7La3Zr2O12")
temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=10)
submit_button = st.button("Submit and Predict")

# === cache predictor loader ===
@st.cache_resource(show_spinner=False)
def load_predictor():
    return TabularPredictor.load("./ag-20251024_075719")

# === MP element colors (subset; add more as needed) ===
MP_ELEMENT_COLORS = {
    "H":"ffffff","He":"d9ffff","Li":"cc80ff","Be":"c2ff00","B":"ffb5b5",
    "C":"909090","N":"3050f8","O":"ff0d0d","F":"90e050","Ne":"b3e3f5",
    "Na":"ab5cf2","Mg":"8aff00","Al":"bfa6a6","Si":"f0c8a0","P":"ff8000",
    "S":"ffff30","Cl":"1ff01f","Ar":"80d1e3","K":"8f40d4","Ca":"3dff00",
    "La":"70d4ff","Zr":"94e0e0"
}
def hex_to_0x(hexstr):
    if hexstr is None: return "0xffffff"
    h = hexstr.lstrip("#")
    if len(h)==6:
        return "0x" + h
    return "0xffffff"

# === Structure retrieval functions (MP then COD fallback) ===
def _to_conventional_safe(s):
    try:
        return s.get_conventional_structure()
    except Exception:
        return s

def _structure_matches_formula(s, formula):
    try:
        expected = {el.symbol for el in Composition(formula).elements}
        present = {el.symbol for el in s.composition.elements}
        return expected.issubset(present)
    except Exception:
        return True

def load_from_MP(query: str):
    try:
        with MPRester(MP_API_KEY) as mpr:
            # if mp-id provided
            if isinstance(query, str) and query.lower().startswith("mp-"):
                try:
                    s = mpr.get_structure_by_material_id(query)
                    return _to_conventional_safe(s)
                except:
                    pass
            # try summary.search
            try:
                results = mpr.summary.search(formula=query)
                if results:
                    for entry in results:
                        s = entry.structure
                        s_conv = _to_conventional_safe(s)
                        if _structure_matches_formula(s_conv, query):
                            return s_conv
                    # fallback
                    return _to_conventional_safe(results[0].structure)
            except Exception:
                pass
            # fallback query/get_entries/get_structures
            try:
                q = mpr.query(criteria={"formula": query}, properties=["material_id"])
                if q:
                    mid = q[0].get("material_id")
                    if mid:
                        s = mpr.get_structure_by_material_id(mid)
                        return _to_conventional_safe(s)
            except:
                pass
            try:
                entries = mpr.get_entries(query)
                if entries:
                    return _to_conventional_safe(entries[0].structure)
            except:
                pass
            try:
                structs = mpr.get_structures(query)
                if structs:
                    return _to_conventional_safe(structs[0])
            except:
                pass
        return None
    except Exception as e:
        st.error(f"Materials Project fetch failed: {e}")
        return None

def load_from_COD(formula: str):
    try:
        url = f"https://www.crystallography.net/cod/result?format=core-formula&q={formula}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split()
        if len(lines)==0:
            return None
        cod_id = lines[0]
        cif_url = f"https://www.crystallography.net/cod/{cod_id}.cif"
        cif_bytes = requests.get(cif_url, timeout=10).content
        return Structure.from_str(cif_bytes.decode(), fmt="cif")
    except Exception:
        return None

def load_crystal_structure_public(query: str):
    st.info("Searching public databases for crystal structure...")
    s = load_from_MP(query)
    if s:
        st.success("Structure found in Materials Project ✓")
        return s
    s = load_from_COD(query)
    if s:
        st.success("Structure found in COD ✓")
        return s
    st.error("No structure found in public databases.")
    return None

# === robust manual wrap (force fractional coords to [0,1)) ===
def wrap_structure_manual(s: Structure):
    lat = s.lattice
    new_species = []
    new_frac = []
    new_props = []
    for site in s.sites:
        fc = np.mod(site.frac_coords, 1.0)
        new_species.append(site.species)
        new_frac.append(fc)
        new_props.append(site.properties if hasattr(site, "properties") else {})
    # Construct Structure from fractional coords
    return Structure(lat, [sp for sp in new_species], [f for f in new_frac], coords_are_cartesian=False, site_properties={"properties": new_props})

# === make supercell helper ===
def make_supercell_safe(s, size=(2,2,2)):
    try:
        sc = s.copy()
        sc.make_supercell(size)
        return sc
    except Exception:
        return s

# === Automatic bond generation using CrystalNN ===
def generate_bonds_crystalnn(s: Structure, cutoff_max=4.5):
    """
    Use CrystalNN to determine neighbors (bonds).
    Returns list of (i, j) pairs (i<j) and list of midpoints/colors/radii for drawing cylinders.
    """
    cnn = CrystalNN()
    bonds = set()
    bond_data = []
    for i, site in enumerate(s.sites):
        try:
            neigh_info = cnn.get_nn_info(s, i)
        except Exception:
            # fallback: simple distance cutoff
            neigh_info = []
            for j, other in enumerate(s.sites):
                if i == j: continue
                d = np.linalg.norm(np.array(site.coords) - np.array(other.coords))
                if d <= cutoff_max:
                    neigh_info.append({"site_index": j})
        for n in neigh_info:
            j = n.get("site_index")
            if j is None: continue
            a,b = min(i,j), max(i,j)
            if a==b: continue
            if (a,b) not in bonds:
                bonds.add((a,b))
                bond_data.append((a,b))
    return bond_data

# === MP-like py3Dmol rendering with bonds ===
def display_structure_mp_like(structure):
    try:
        # 1) initial wrap
        try:
            s = wrap_structure_manual(structure)
        except Exception:
            s = structure.copy()
        # 2) orthogonalize initial
        try:
            s = s.get_orthogonalized_structure()
        except Exception:
            pass
        # 3) supercell
        s = make_supercell_safe(s, (2,2,2))
        # 4) wrap again (critical)
        s = wrap_structure_manual(s)
        # 5) orthogonalize again
        try:
            s = s.get_orthogonalized_structure()
        except Exception:
            pass

        # Export for base model
        cif_str = s.to(fmt="cif")
        view = py3Dmol.view(width=800, height=620)
        view.addModel(cif_str, "cif")

        # Style atoms (sphere + stick base)
        for site in s:
            elem = site.specie.symbol
            try:
                cov = Element(elem).covalent_radius or 0.77
            except Exception:
                cov = 0.77
            sphere_scale = max(0.15, cov * 0.45)
            color = hex_to_0x(MP_ELEMENT_COLORS.get(elem, None))
            view.setStyle({"elem": elem}, {"sphere": {"scale": sphere_scale, "color": color},
                                          "stick": {"radius": 0.10, "color": color}})

        # Add unit cell / camera
        view.addUnitCell({"color":"0x000000","linewidth":1.2})
        view.setBackgroundColor("white")
        view.setProjection("orthographic")
        view.zoomTo()

        # 6) Generate bonds and draw them as cylinders (so bonds are visible)
        bond_pairs = generate_bonds_crystalnn(s)
        # Add cylinder for each bond
        for (i,j) in bond_pairs:
            a = s.sites[i].coords
            b = s.sites[j].coords
            # bond color: average of two atoms or gray
            elem_a = s.sites[i].specie.symbol
            elem_b = s.sites[j].specie.symbol
            col_a = MP_ELEMENT_COLORS.get(elem_a, "ffffff")
            col_b = MP_ELEMENT_COLORS.get(elem_b, "ffffff")
            # choose midpoint color as average (simple heuristic)
            try:
                ca = int(col_a, 16)
                cb = int(col_b, 16)
                # average rgb
                ra = (ca >> 16) & 0xFF; ga = (ca >> 8) & 0xFF; ba = ca & 0xFF
                rb = (cb >> 16) & 0xFF; gb = (cb >> 8) & 0xFF; bb = cb & 0xFF
                rc = (ra+rb)//2; gc_ = (ga+gb)//2; bc = (ba+bb)//2
                col_hex = "0x{:02x}{:02x}{:02x}".format(rc, gc_, bc)
            except Exception:
                col_hex = "0x888888"
            # py3Dmol: use addCylinder for bond rendering
            try:
                view.addCylinder({
                    "start": {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
                    "end":   {"x": float(b[0]), "y": float(b[1]), "z": float(b[2])},
                    "radius": 0.08,
                    "color": col_hex,
                    "fromCap": 0, "toCap": 0
                })
            except Exception:
                # if cylinder fails, ignore
                pass

        # 7) Optional: polyhedra approx (scipy ConvexHull) - can be heavy; silence failures
        if _HAVE_SCIPY:
            try:
                # build small polyhedra for cations only (e.g., Zr, La)
                for i, site in enumerate(s.sites):
                    celem = site.specie.symbol
                    if celem in ("Zr","La","Ti","Fe","Mn","Co","Ni","Cu"):
                        # collect neighbors within cov radius + tolerance
                        neigh_coords = []
                        for j, other in enumerate(s.sites):
                            if i==j: continue
                            try:
                                r1 = Element(celem).covalent_radius or 0.7
                            except:
                                r1 = 0.7
                            try:
                                r2 = Element(other.specie.symbol).covalent_radius or 0.7
                            except:
                                r2 = 0.7
                            cutoff = 1.2 * (r1 + r2)
                            dist = np.linalg.norm(np.array(site.coords) - np.array(other.coords))
                            if dist <= cutoff:
                                neigh_coords.append(other.coords)
                        if len(neigh_coords) < 4:
                            continue
                        pts = np.vstack(neigh_coords)
                        hull = ConvexHull(pts)
                        # build obj text
                        obj_lines = []
                        for p in pts:
                            obj_lines.append("v {:.6f} {:.6f} {:.6f}".format(p[0], p[1], p[2]))
                        for face in hull.simplices:
                            a,b,c = face
                            obj_lines.append("f {} {} {}".format(a+1,b+1,c+1))
                        obj_text = "\n".join(obj_lines)
                        try:
                            view.addModel(obj_text, "obj")
                            # mesh style for last model
                            view.setStyle({"model": -1}, {"mesh": {"color":"0x8fbc8f", "opacity":0.32}})
                        except:
                            pass
            except Exception:
                pass

        # Render HTML
        st.components.v1.html(view._make_html(), height=700)

    except Exception as e:
        st.error(f"MP-like visualization failed: {e}")

# === Feature extraction (your original) ===
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

# === descriptors used ===
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

# === Main flow ===
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula or MP id.")
        st.stop()

    st.subheader("Crystal Structure (MP-like rendering)")
    structure = load_crystal_structure_public(formula_input)
    if structure:
        display_structure_mp_like(structure)
    else:
        st.warning("Cannot find structure for this material.")

    # Features + prediction
    with st.spinner("Extracting features and predicting..."):
        features = calculate_material_features(formula_input)
        st.write(f"Extracted {len(features)} features.")
        selected = filter_selected_features(features, required_descriptors, temperature)
        st.subheader("Selected Features")
        st.dataframe(pd.DataFrame([selected]))

    st.subheader("Prediction Results")
    try:
        predictor = load_predictor()
    except Exception:
        predictor = None
        st.error("Model loading failed.")
    if predictor:
        input_data = {"Formula": [formula_input], "Temp": [temperature]}
        for f in required_descriptors:
            if f != "Temp":
                input_data[f] = [features.get(f, 0.0)]
        input_df = pd.DataFrame(input_data)
        models = ["CatBoost","ExtraTreesMSE","LightGBM","KNeighborsDist","WeightedEnsemble_L2","XGBoost"]
        results = {}
        for model in models:
            try:
                results[model] = predictor.predict(input_df, model=model)
            except Exception:
                results[model] = "Error"
        st.dataframe(pd.DataFrame(results).iloc[:1,:])
        del predictor
        gc.collect()
