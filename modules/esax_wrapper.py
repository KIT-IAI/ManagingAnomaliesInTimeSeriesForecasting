import numpy as np
import pandas as pd
import xarray as xr


def prepare_results(found_motifs, startpoints, indexes_subs):
    """
    This method applies the eSAX method and transforms its output into xr.DataArrays
    @param found_motifs: result dict of eSAX
    @param startpoints: startpoints of all the subsequences
    @param indexes_subs: list with sequences of timestamps
    @return: dict with xr.DataArrays: subsequences, SAX dataframe, motifs (symbolic, non-symbolic), collision matrix,
    indices where the motifs start, non-symbolic subsequences and motif_raw as a list of lists
    """
    indexes = found_motifs['indexes']
    ts_sax_df = found_motifs['ts_sax_df']
    sequences = found_motifs['ts_subs']
    col_mat = found_motifs['col_mat']

    motif_raw = []
    motif_sax = []

    for val in indexes:
        motif_raw_indices = np.where(np.isin(ts_sax_df.index, list(val)))[0]
        motif_raw.append([sequences[v] for v in motif_raw_indices])
        motif_sax.append(ts_sax_df.iloc[motif_raw_indices, :])

    longest_seq = max([len(seq) for seq in sequences])
    sequences_dataarray = np.full((len(sequences), longest_seq), np.nan)

    # NOTE: time axis of the individual subsequences is lost in this step
    for idx, val in enumerate(sequences):
        for idx2, val2 in enumerate(val):
            sequences_dataarray[idx, idx2] = val2

    sequences_dataarray = xr.DataArray(sequences_dataarray, coords={
        'startpoint': ('startpoint', startpoints),
        'index': ('index', indexes_subs[0])}, dims=['startpoint', 'index'])
    ts_sax_dataarray = xr.DataArray(ts_sax_df, dims=['StartP', 'y'])

    # The counter is used to determine how many sequences are in each motif
    # Insert raw motifs into a dataset
    number_of_motifs = len(motif_raw)
    counter = 0
    number_of_sequences = []
    sequences_list = []
    for i in motif_raw:
        for j in i:
            counter += 1
            sequences_list.append(j)
        number_of_sequences.append(counter)
        counter = 0

    indexcol = np.repeat(range(0, number_of_motifs), number_of_sequences, axis=0)

    # Convert collision matrix into a dataset
    col_mat_dataarray = xr.DataArray(col_mat, dims=['x', 'y'])

    # Skip if no motifs are found
    if len(sequences_list) == 0:
        return None

    longest_mot = max([len(seq) for seq in sequences_list])
    motifs_raw_dataarray = np.full((len(indexcol), longest_mot), np.nan)
    for idx, _ in enumerate(sequences_list):
        for idx2, val2 in enumerate(_):
            motifs_raw_dataarray[idx, idx2] = val2

    motifs_raw_dataarray = pd.DataFrame(motifs_raw_dataarray)
    motifs_raw_dataarray.insert(0, 'motif', indexcol)
    motifs_raw_dataarray = motifs_raw_dataarray.set_index('motif')

    # Add arbitrary timestamps for the following steps

    if len(sequences) != 0:
        motifs_raw_dataarray = xr.DataArray(motifs_raw_dataarray, dims=['motif', 'index'],
                                            coords={'motif': indexcol, 'index': indexes_subs[0]})
    else:
        motifs_raw_dataarray = xr.DataArray(motifs_raw_dataarray, dims=['motif', 'index'],
                                            coords={'motif': indexcol})

    # Insert sax represented motifs into a dataset

    motif_sax_df = pd.concat([motif for motif in motif_sax], 0)
    motif_sax_df['motif'] = indexcol
    motif_sax_df = motif_sax_df.set_index('motif')

    motifs_sax_dataarray = xr.DataArray(motif_sax_df, dims=['motif', 'y'])

    # Convert the start indices of the motifs into a dataset
    ind_list = []
    for i in indexes:
        for j in i:
            ind_list.append(j)

    ind_df = pd.DataFrame(ind_list)
    ind_df.insert(1, 'motif', indexcol)
    ind_df = ind_df.set_index('motif')
    ind_df = xr.DataArray(ind_df, dims=['motif', 'index'])

    return {'ts_subs': sequences_dataarray,
            'ts_sax': ts_sax_dataarray,
            'motifs_raw': motifs_raw_dataarray,
            'motifs_raw_list': motif_raw,
            'motifs_sax': motifs_sax_dataarray,
            'col_mat': col_mat_dataarray,
            'indices': ind_df}