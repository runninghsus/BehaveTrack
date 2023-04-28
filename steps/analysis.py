import streamlit as st
import numpy as np
from utils.import_utils import *
from utils.visuals_utils import *


def main():
    st.markdown(f" <h1 style='text-align: left; color: #FF6A95; font-size:30px; "
                f"font-family:Avenir; font-weight:normal;'>New data upload</h1> "
                , unsafe_allow_html=True)
    st.write('')
    # bsoid_t, asoid_t = st.tabs(['B-SOiD', 'A-SOiD'])
    try:
        if 'bsoid_classifier' in st.session_state:
            st.session_state['classifier'] = st.session_state['bsoid_classifier']
            st.write('B-SOiD classifier loaded!')
        elif 'asoid_classifier' in st.session_state:
            st.session_state['classifier'] = st.session_state['asoid_classifier']
            st.write('A-SOiD classifier loaded!')
    except:
        st.warning('please return to home to upload sav files')
    try:
        conditions_list = list(st.session_state['features'].keys())
        text_ = f":orange[**RESET**] data from conditions: {' & '.join([i.rpartition('_')[2] for i in conditions_list])}!"

        def clear_data():
            del st.session_state['features']
            del st.session_state['pose']
            del st.session_state['bodypart_names']
            del st.session_state['bodypart_idx']

        st.button(text_, on_click=clear_data)

    except:
        num_cond = st.number_input('How many conditions?', min_value=2, max_value=10, value=2)
        uploaded_files = {f'condition_{key}': [] for key in range(num_cond)}
        pose = {f'condition_{key}': [] for key in range(num_cond)}
        features = {f'condition_{key}': [] for key in range(num_cond)}
        condition_prompt(uploaded_files, num_cond)
        try:
            data_raw = []
            for i, condition in enumerate(list(uploaded_files.keys())):
                placeholder = st.empty()
                data_raw.append(read_csvfiles(uploaded_files[condition]))
                if i == 0:
                    p, pose_chosen = get_bodyparts(placeholder, data_raw[i])
                    if 'bodypart_names' not in st.session_state or 'bodypart' not in st.session_state:
                        st.session_state['bodypart_names'] = p
                        st.session_state['bodypart_idx'] = pose_chosen
            conditions_list = list(uploaded_files.keys())
            if st.button(f"extract features from conditions: "
                         f"{' & '.join([i.rpartition('_')[2] for i in conditions_list])}"):
                for i, condition in enumerate(
                        stqdm(list(uploaded_files.keys()),
                              desc=f"Extracting spatiotemporal features from "
                                   f"{' & '.join([i.rpartition('_')[2] for i in conditions_list])}")):
                    loader = csv_upload(data_raw[i], pose_chosen, condition, framerate=30)
                    pose[condition], features[condition] = loader.main()
                if 'pose' not in st.session_state:
                    st.session_state['pose'] = pose
                if 'features' not in st.session_state:
                    st.session_state['features'] = features
                st.markdown(f":blue[saved features] from conditions: "
                            f":orange[{' & '.join([i.rpartition('_')[2] for i in conditions_list])}!]")
        except:
            pass
            if 'features' in st.session_state:
                st.experimental_rerun()
    st.divider()

    try:
        if 'pose' in st.session_state:
            mid_expander = st.expander('Analysis method', expanded=True)
            analysis_chosen = mid_expander.radio('',
                                                 ['ethogram', 'behavioral location', 'behavioral ratio',
                                                  'frequency',
                                                  'duration', 'transition', 'kinematics'],
                                                 horizontal=True)
            if analysis_chosen == 'ethogram':
                condition_etho_plot()
            if analysis_chosen == 'behavioral ratio':
                condition_pie_plot()
            if analysis_chosen == 'behavioral location':
                condition_location_plot()
            if analysis_chosen == 'frequency':
                condition_bar_plot()
            if analysis_chosen == 'duration':
                condition_ridge_plot()
            if analysis_chosen == 'transition':
                condition_transmat_plot()
            # if analysis_chosen == 'kinematics':
            #     condition_kinematix_plot()
    except:
        pass

