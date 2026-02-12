"""
EMGlab format export.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np


def find_max_channel_for_unit(
    timestamps: np.ndarray,
    emg_data: np.ndarray,
    fsamp: int
) -> int:
    """Find the channel with maximum amplitude for a unit."""
    window = int(0.04 * fsamp)
    hw = window // 2
    
    emg_windows = []
    for ts in timestamps:
        ts = int(ts)
        if ts - hw >= 0 and ts + hw < emg_data.shape[1]:
            emg_windows.append(emg_data[:, ts - hw:ts + hw])
    
    if len(emg_windows) >= 5:
        sta = np.mean(emg_windows, axis=0)
        peak_to_peak = np.ptp(sta, axis=1)
        return int(np.argmax(peak_to_peak))
    
    return 0


def export_to_emglab(
    filepath: Path,
    timestamps_list: List[np.ndarray],
    emg_data: Optional[np.ndarray],
    fsamp: int
) -> None:
    """
    Export decomposition to EMGlab .eaf format.
    
    Parameters
    ----------
    filepath : Path
        Output file path
    timestamps_list : List[np.ndarray]
        List of timestamp arrays for each unit
    emg_data : np.ndarray or None
        EMG data for finding max channels [channels x samples]
    fsamp : int
        Sampling frequency
    """
    # Find max channels for each unit
    max_channels = []
    for unit_idx, timestamps in enumerate(timestamps_list):
        if len(timestamps) >= 5 and emg_data is not None:
            max_ch = find_max_channel_for_unit(timestamps, emg_data, fsamp)
        else:
            max_ch = 0
        max_channels.append(max_ch)
    
    unique_channels = sorted(set(max_channels))
    
    # Build data entries
    data = []
    for unit_idx, timestamps in enumerate(timestamps_list):
        if len(timestamps) == 0:
            continue
        
        timestamps_sec = timestamps / fsamp
        
        for channel in unique_channels:
            unit_data = [
                (f'{ts:.8f}', str(unit_idx + 1), str(channel + 1))
                for ts in timestamps_sec
            ]
            data.extend(unit_data)
    
    # Sort by time
    sorted_data = sorted(data, key=lambda x: float(x[0]))
    data_str = '\n'.join(' '.join(item) for item in sorted_data)
    
    # EAF template
    eaf_content = f"""<?xml version="1.0" encoding="ASCII"?>
<emglab_annotation_file
xmlns="http://ece.wpi.edu/~ted"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://ece.wpi.edu/~ted http://ece.wpi.edu/~ted/emglab_annotation_file.xsd">
<emglab_version>0.01</emglab_version>
<emglab_spike_header>
<time></time>
<unit></unit>
<chan></chan>
</emglab_spike_header>
<emglab_spike_events>
{data_str}
</emglab_spike_events>
</emglab_annotation_file>"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(eaf_content)