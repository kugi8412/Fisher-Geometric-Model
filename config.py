import os
import time


def delete_temp_files(temp_dir: str):
    """ Safe deletion of temporary files.
    """
    max_retries = 2
    for i in range(max_retries):
        try:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(temp_dir)
            break
        except PermissionError:
            time.sleep(0.2 * (i+1)) # waiting to close all files
        except FileNotFoundError:
            break

''' The method served to parallelize multiple computations on a CUDA device.
It does not work for the CPU and returns numerous warnings, but returns correct results.
An alternative import can be made, or the relevant functions can be replaced in the analyses.py file.

import torch
import contextlib
import numpy as np
import streamlit as st
import plotly.express as px
import torch.multiprocessing as mp
import streamlit.runtime.scriptrunner as scriptrunner
from typing import Dict
from analyses import run_simulation

@contextlib.contextmanager
def suppress_streamlit():
    original_ctx = scriptrunner.get_script_run_ctx
    scriptrunner.get_script_run_ctx = lambda: None
    try:
        yield
    finally:
        scriptrunner.get_script_run_ctx = original_ctx

def init_device_process():
    """ Initialization process with CUDA.
    """
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()

def parallel_wrapper(args):
    with suppress_streamlit():
        try:
            params, param1, param2, v1, v2, trials, i, j = args
            device = torch.device('cuda')
            
            # Conversion to correct device
            converted_params = {}
            for k, v in params.items():
                if isinstance(v, torch.Tensor):
                    converted_params[k] = v.to(device, non_blocking=True) if device.type == 'cuda' else v.cpu()
                else:
                    converted_params[k] = v
            
            survivals = []
            for _ in range(trials):
                with torch.cuda.device(device):
                    population_history = run_simulation(
                        converted_params, None, None, None, analysis=True
                    )
                survivals.append(population_history[-1] if population_history else 0)

            # Memory cleaning
            del converted_params
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return (i, j, np.mean(survivals), v1, v2)
            
        except Exception as e:
            print(f"Error in process: {str(e)}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return (i, j, 0, v1, v2)

def run_parameter_analysis(base_params: Dict[str, float],
                           param1: str, 
                           param2: str,
                           grid_size1: int,
                           grid_size2: int,
                           trials: int):
    """Improved parameter analysis function
    to create heatmap of mean survivors.
    """
    
    # Multiprocessing
    mp.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    value_ranges = {
        param1: np.linspace(0.1, 2.0, grid_size1) if param1 in ["radius_multiplier", "selection"] 
                else np.linspace(0.0, 1.0, grid_size1),
        param2: np.linspace(0.01, 1.0, grid_size2)
    }

    tasks = []
    for i, v1 in enumerate(value_ranges[param1]):
        for j, v2 in enumerate(value_ranges[param2]):
            params = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v 
                for k, v in base_params.items()
            }
            params.update({param1: v1, param2: v2})
            tasks.append((params, param1, param2, v1, v2, trials, i, j))

    results = np.zeros((grid_size1, grid_size2))
    total = len(tasks)

    with st.expander("Progress", expanded=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        last_update = time.time()

        context = mp.get_context('spawn')
        with context.Pool(
            processes=mp.cpu_count(),
            initializer=init_device_process
        ) as pool:
            for idx, (i, j, mean, v1, v2) in enumerate(pool.imap_unordered(parallel_wrapper, tasks)):
                results[i, j] = mean
                
                # Actualization of interface
                if time.time() - last_update > 0.2:
                    progress = (idx + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Progress: {idx+1}/{total}\n"
                        f"params: {param1}={v1:.2f}, {param2}={v2:.2f}\n"
                        f"Mean: {mean:.2f}"
                    )
                    last_update = time.time()

    # Generating Plot
    fig = px.imshow(
        results,
        x=value_ranges[param2],
        y=value_ranges[param1],
        labels={'x': param2, 'y': param1},
        color_continuous_scale='Inferno',
        aspect='auto'
    )
    fig.update_layout(
        title=f"Average Number of Survivors: {param1} vs {param2}",
        xaxis_title=param2,
        yaxis_title=param1
    )
    
    st.plotly_chart(fig)
    st.success("The analysis is complete!")
 '''
