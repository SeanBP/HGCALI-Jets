import uproot as ur
import awkward as ak
import pandas as pd
import itertools
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import fastjet
import matplotlib.colors as mcolors
import os

# ============================================================
# Collect ROOT files
# ============================================================
R=1.0
base_dir = "/media/miguel/Expansion/ePIC_feb2025_campaign/NC_DIS_18x275/"

root_files = [
    os.path.join(dirpath, filename)
    for dirpath, _, filenames in os.walk(base_dir)
    for filename in filenames
    if filename.endswith(".root")
]

print(f"Found {len(root_files)} ROOT files in {base_dir}")




# ============================================================
# Helper Functions
# ============================================================

def rotateY(xdata, zdata, angle):
    s = np.sin(angle)
    c = np.cos(angle)
    rotatedz = c * zdata - s * xdata
    rotatedx = s * zdata + c * xdata
    return rotatedx, rotatedz


def eta(px, py, pz):
    pT = np.sqrt(px**2 + py**2)
    theta = np.arctan2(pT, pz)
    return -np.log(np.tan(theta / 2))


def phi(x, y):
    return np.arctan2(y, x)


def momentum_xyz(px, py, pz):
    norm = np.sqrt(px**2 + py**2 + pz**2)
    x = px / norm
    y = py / norm
    z = pz / norm
    return x, y, z

# ============================================================
# Helper Functions
# ============================================================
def configure_plotting():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'


def rotateY(xdata, zdata, angle):
    s = np.sin(angle)
    c = np.cos(angle)
    rotatedz = c * zdata - s * xdata
    rotatedx = s * zdata + c * xdata
    return rotatedx, rotatedz


def eta(px, py, pz):
    pT = np.sqrt(px**2 + py**2)
    theta = np.arctan2(pT, pz)
    return -np.log(np.tan(theta / 2))


def phi(x, y):
    return np.arctan2(y, x)


def momentum_xyz(px, py, pz):
    norm = np.sqrt(px**2 + py**2 + pz**2)
    x = px / norm
    y = py / norm
    z = pz / norm
    return x, y, z


def jet_dict_ak(px, py, pz, e, minE, minHitE=0.0, etaMin=2.5, etaMax=4):
    # --- Preselection ---
    hit_eta = eta(px, py, pz)
    mask = (e >= minHitE)
    px, py, pz, e = px[mask], py[mask], pz[mask], e[mask]

    # --- Build momentum records ---
    array = ak.zip(
        {"px": px, "py": py, "pz": pz, "E": e},
        with_name="Momentum4D",
        behavior=ak.behavior
    )

    # --- Jet clustering ---
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    cluster = fastjet.ClusterSequence(array, jetdef)

    # Get all jets (no pT cut yet)
    jets = cluster.inclusive_jets()
    constituents = cluster.constituents()

    # --- Compute pt for jets ---
    jets["pt"] = np.sqrt(jets["px"] ** 2 + jets["py"] ** 2)

    # --- Apply cuts ---
    jeta = eta(jets.px, jets.py, jets.pz)
    jphi = phi(jets.px, jets.py)
    cut = (
        (jets.E >= minE)
        & (jeta >= etaMin)
        & (jeta <= etaMax)
    )

    # Apply the same mask to jets and constituents
    jets = jets[cut]
    constituents = constituents[cut]
    jeta = eta(jets.px, jets.py, jets.pz)
    jphi = phi(jets.px, jets.py)
    # --- Return full dict ---
    return {
        "energy": jets.E,
        "eta": jeta,
        "phi": jphi,
        "pt": jets.pt,
        "constituents": constituents
    }



def jets_to_df(jets_dict, start_event=0):
    # Ensure we have something to process
    if any(len(jets_dict[key]) == 0 for key in ['energy', 'eta', 'phi']):
        return pd.DataFrame(columns=['energy', 'eta', 'phi', 'event', 'constituents'])

    num_events = len(jets_dict['energy'])
    if num_events == 0:
        return pd.DataFrame(columns=['energy', 'eta', 'phi', 'event', 'constituents'])

    # Generate event numbers aligned to each jet
    event_numbers = ak.concatenate(
        [ak.full_like(jets_dict['energy'][i], start_event + i, dtype=int) for i in range(num_events)]
    )

    # Flatten per-jet data
    energy_flat = ak.flatten(jets_dict['energy'])
    eta_flat = ak.flatten(jets_dict['eta'])
    phi_flat = ak.flatten(jets_dict['phi'])

    # --- Flatten constituents ---
    # Each entry in jets_dict['constituents'] corresponds to a jet → a list of particles
    # We'll convert each jet’s constituent list into a JSON string for DataFrame storage
    constituent_list = []
    if "constituents" in jets_dict:
        for jet_const in ak.flatten(jets_dict["constituents"]):
            if len(jet_const) == 0:
                constituent_list.append(None)
            else:
                # Extract px, py, pz, E into a serializable structure
                const_px = ak.to_list(jet_const.px)
                const_py = ak.to_list(jet_const.py)
                const_pz = ak.to_list(jet_const.pz)
                const_E  = ak.to_list(jet_const.E)
                constituent_list.append({
                    "px": const_px,
                    "py": const_py,
                    "pz": const_pz,
                    "E":  const_E
                })
    else:
        constituent_list = [None] * len(energy_flat)

    # --- Build DataFrame ---
    return pd.DataFrame({
        'energy': energy_flat.to_numpy(),
        'eta': eta_flat.to_numpy(),
        'phi': phi_flat.to_numpy(),
        'event': event_numbers.to_numpy(),
        'constituents': constituent_list
    })



# ============================================================
# Main Jet Matching Loop
# ============================================================
truth_dfs = []
reco_dfs = []
truth_constituents = []
reco_constituents = []

event_counter = 0

cluster_names = [
    "LFHCALClusters/LFHCALClusters",
    "EcalEndcapPClusters/EcalEndcapPClusters",
    "EcalEndcapPInsertClusters/EcalEndcapPInsertClusters",
    "HcalEndcapPInsertClusters/HcalEndcapPInsertClusters"
]

for idx, fpath in enumerate(root_files, 1):


    print(f"[{idx}/{len(root_files)}] Processing {os.path.basename(fpath)}")


    try:
        events = ur.open(fpath)["events"]
        # --- Truth momenta ---
        gen_status = events["MCParticles/MCParticles.generatorStatus"].array(library="ak")

        px_truth = events["MCParticles/MCParticles.momentum.x"].array(library="ak")
        py_truth = events["MCParticles/MCParticles.momentum.y"].array(library="ak")
        pz_truth = events["MCParticles/MCParticles.momentum.z"].array(library="ak")
        mass_truth = events["MCParticles/MCParticles.mass"].array(library="ak")

        px_rot, pz_rot = rotateY(px_truth, pz_truth, 0.025)
        e_truth = np.sqrt(px_rot**2 + py_truth**2 + pz_rot**2 + mass_truth**2)

        mask = (gen_status == 1)
        px_rot, py_truth, pz_rot, e_truth = [
            arr[mask] for arr in (px_rot, py_truth, pz_rot, e_truth)
        ]

        truth_jets = jet_dict_ak(
            px_rot, py_truth, pz_rot, e_truth,
            minHitE=0.0, minE=0, etaMin=3.0, etaMax=4.0
        )

        # --- Reco ---
        x_branches = [events[f"{name}.position.x"].array(library="ak") for name in cluster_names]
        y_branches = [events[f"{name}.position.y"].array(library="ak") for name in cluster_names]
        z_branches = [events[f"{name}.position.z"].array(library="ak") for name in cluster_names]
        e_branches = [events[f"{name}.energy"].array(library="ak")      for name in cluster_names]

        x_all = ak.concatenate(x_branches, axis=1)
        y_all = ak.concatenate(y_branches, axis=1)
        z_all = ak.concatenate(z_branches, axis=1)
        e_all = ak.concatenate(e_branches, axis=1)

        norms = np.sqrt(x_all**2 + y_all**2 + z_all**2)
        px_reco = (x_all / norms) * e_all
        py_reco = (y_all / norms) * e_all
        pz_reco = (z_all / norms) * e_all

        px_rot, pz_rot = rotateY(ak.Array(px_reco), ak.Array(pz_reco), 0.025)

        reco_jets = jet_dict_ak(
            ak.Array(px_rot), ak.Array(py_reco), ak.Array(pz_rot), ak.Array(e_all),
            minHitE=1.5, minE=0, etaMin=3, etaMax=4.0
        )

        # Convert to DataFrames
        truth_df = jets_to_df(truth_jets, start_event=event_counter)
        reco_df = jets_to_df(reco_jets, start_event=event_counter)

        truth_dfs.append(truth_df)
        reco_dfs.append(reco_df)

        event_counter += len(truth_jets['energy'])


    except Exception as e:
        print(f"⚠️ Error processing events in file index {idx}: {e}")
        continue

# ============================================================
# Matching
# ============================================================
truth_df = pd.concat(truth_dfs, ignore_index=True)
reco_df = pd.concat(reco_dfs, ignore_index=True)

reco_df["reco_idx"] = reco_df.groupby("event").cumcount()
truth_df["truth_idx"] = truth_df.groupby("event").cumcount()

pairs = reco_df.merge(truth_df, on="event", suffixes=("_reco", "_truth"))
pairs["deta"] = pairs["eta_reco"] - pairs["eta_truth"]
pairs["dphi"] = (pairs["phi_reco"] - pairs["phi_truth"] + np.pi) % (2 * np.pi) - np.pi
pairs["deltaR"] = np.sqrt(pairs["deta"] ** 2 + pairs["dphi"] ** 2)

closest_pairs = (
    pairs.loc[pairs.groupby(["event", "reco_idx"])["deltaR"].idxmin()]
    .sort_values(["event", "deltaR"])
    .drop_duplicates(subset=["event", "truth_idx"], keep="first")
    .drop_duplicates(subset=["event", "reco_idx"], keep="first")
)
closest_pairs = closest_pairs[closest_pairs["deltaR"] <= R/2]

paired_reco = set(zip(closest_pairs["event"], closest_pairs["reco_idx"]))
paired_truth = set(zip(closest_pairs["event"], closest_pairs["truth_idx"]))

unpaired_reco_mask = ~reco_df.apply(lambda row: (row["event"], row["reco_idx"]) in paired_reco, axis=1)
unpaired_truth_mask = ~truth_df.apply(lambda row: (row["event"], row["truth_idx"]) in paired_truth, axis=1)

unpaired_reco = pd.DataFrame(reco_df[unpaired_reco_mask])
unpaired_truth = pd.DataFrame(truth_df[unpaired_truth_mask])

# ============================================================
# Save Output
# ============================================================
data_dict = {
    "closest_pairs": closest_pairs,
    "unpaired_reco": unpaired_reco,
    "unpaired_truth": unpaired_truth
}

pd.to_pickle(data_dict, "jets_R1_E0_halfR.pkl")
print(f"✅ Finished. Total events processed: {event_counter}")
