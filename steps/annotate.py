# from app import swap_app
import base64
import glob
import io
import os

import numpy as np
import pandas as pd
import streamlit as st


def prompt_setup(framerate, working_dir, prefix):
    left_col, right_col = st.columns(2)
    left_expand = left_col.expander('Select a video file:', expanded=True)
    right_expand = right_col.expander('Select the corresponding pose file:', expanded=True)
    num_outliers = None
    output_fps = None
    frame_dir = None
    shortvid_dir = None
    new_pose_list = None

    new_videos = left_expand.file_uploader('Upload video files',
                                           accept_multiple_files=False,
                                           type=['avi', 'mp4'], key='video')
    new_pose_csvs = [right_expand.file_uploader('Upload corresponding pose csv files',
                                                accept_multiple_files=False,
                                                type='csv', key='pose')]
    try:
        new_pose_list = []
        for i, f in enumerate(new_pose_csvs):
            current_pose = pd.read_csv(new_pose_csvs[i],
                                       header=[0, 1, 2], sep=",", index_col=0)
            idx_selected = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
            new_pose_list.append(np.array(current_pose.iloc[:, idx_selected]))
        col1, col3 = st.columns(2)
        col1_exp = col1.expander('Parameters'.upper(), expanded=True)
        col3_exp = col3.expander('Output folders'.upper(), expanded=True)
        num_outliers = col1_exp.number_input('Number of potential outliers to refine',
                                             min_value=10, max_value=None, value=20)
        output_fps = col1_exp.number_input('Video playback fps',
                                           min_value=1, max_value=None, value=5)
        col1_exp.write(f'equivalent to {round(output_fps / framerate, 2)} X speed')
        frame_dir = col3_exp.text_input('Enter a directory for frames',
                                        os.path.join(working_dir, prefix, new_videos.name.rpartition('.mp4')[0],
                                                     'pngs'),
                                        )

        try:
            os.listdir(frame_dir)
            col3_exp.success(f'Entered **{frame_dir}** as the frame directory.')
        except FileNotFoundError:
            if col3_exp.button('create frame directory'):
                os.makedirs(frame_dir, exist_ok=True)
                st.experimental_rerun()

        shortvid_dir = col3_exp.text_input('Enter a directory for refined videos',
                                           os.path.join(working_dir, prefix, new_videos.name.rpartition('.mp4')[0],
                                                        'refine_vids'),
                                           )
        try:
            os.listdir(shortvid_dir)
            col3_exp.success(f'Entered **{shortvid_dir}** as the refined video directory.')
        except FileNotFoundError:
            if col3_exp.button('create refined video directory'):
                os.makedirs(shortvid_dir, exist_ok=True)
                st.experimental_rerun()
    except:
        pass
    return new_videos, new_pose_list, num_outliers, output_fps, frame_dir, shortvid_dir


def main(config=None):
    st.markdown("""---""")

    [new_videos, new_pose_list, num_outliers, output_fps, frame_dir, shortvid_dir] = \
        prompt_setup(framerate, working_dir, prefix)
    if 'refinements' not in st.session_state:
        st.session_state['refinements'] = {key:
                                               {k: {'choice': None, 'submitted': False}
                                                for k in range(num_outliers)}
                                           for key in annotation_classes}
    # st.write(st.session_state['refinements'])
    # for b_chosen in list(user_choices.keys()):
    #     user_choices[b_chosen] = {key: [] for key in range(num_outliers)}
    if new_videos is not None and len(new_pose_list) > 0:
        if os.path.exists(new_videos.name):
            temporary_location = f'{new_videos.name}'
        else:
            g = io.BytesIO(new_videos.read())  # BytesIO Object
            temporary_location = f'{new_videos.name}'
            with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            out.close()
        if os.path.exists(frame_dir):
            framedir_ = os.listdir(frame_dir)
            if len(framedir_) < 2:
                frame_extraction(video_file=temporary_location, frame_dir=frame_dir)
            else:
                if os.path.exists(shortvid_dir):
                    viddir_ = os.listdir(shortvid_dir)
                    if len(viddir_) < 2:
                        frames2integ = round(float(framerate) * (duration_min / 0.1))
                        [_, _, scalar, _] = load_features(working_dir, prefix)
                        [iterX_model, _, _, _, _, _] = load_iterX(working_dir, prefix)
                        create_videos(new_pose_list, scalar, iterX_model, framerate, frames2integ,
                                      num_outliers, output_fps, annotation_classes,
                                      frame_dir=frame_dir, shortvid_dir=shortvid_dir)

                    else:
                        col_option, col_msg = st.columns(2)
                        # col_msg.success('refinement candidates have been saved!')
                        if col_option.checkbox('Redo? Uncheck after check to prevent from auto-clearing',
                                               False,
                                               key='vr'):
                            try:
                                for file_name in glob.glob(shortvid_dir + "/*"):
                                    os.remove(file_name)
                            except:
                                pass

                        behav_choice = st.selectbox("Select the behavior: ", annotation_classes,
                                                    index=int(0),
                                                    key="behavior_choice")
                        checkbox_autofill = st.checkbox('autofill')
                        alltabs = st.tabs([f'{i}' for i in range(num_outliers)])

                        # if not st.session_state['refinements'][behav_choice]:
                        # st.write(st.session_state['refinements'])

                        for i, tab_ in enumerate(alltabs):
                            with tab_:
                                colL, colR = st.columns([3, 1])
                                file_ = open(
                                    os.path.join(shortvid_dir, f'behavior_{behav_choice}_example_{i}.gif'),
                                    "rb")
                                contents = file_.read()
                                data_url = base64.b64encode(contents).decode("utf-8")
                                file_.close()
                                colL.markdown(
                                    f'<img src="data:image/gif;base64,{data_url}" alt="gif">',
                                    unsafe_allow_html=True,
                                )
                                # st.write([annotation_classes[i] for i in range(len(annotation_classes))],
                                #          'hello')
                                with colR.form(key=f'form_{i}'):
                                    returned_choice = st.radio("Select the correct class: ",
                                                               annotation_classes,
                                                               index=annotation_classes.index(behav_choice),
                                                               key="radio_{}".format(i))

                                    # st.session_state['refinements'][behav_choice][i]["submitted"] = \
                                    #     st.form_submit_button("Submit",
                                    #                           "Press to confirm your choice")
                                    if st.form_submit_button("Submit",
                                                             "Press to confirm your choice"):
                                        # st.write('hello')
                                        st.session_state['refinements'][behav_choice][i]["submitted"] = True
                                        st.session_state['refinements'][behav_choice][i][
                                            "choice"] = returned_choice
                                    # else:
                                    #     st.experimental_rerun()

                                    # if st.session_state['refinements'][behav_choice][i]["submitted"] == True:

                                    if checkbox_autofill:
                                        if st.session_state['refinements'][behav_choice][i]["submitted"] == False:
                                            st.session_state['refinements'][behav_choice][i][
                                                "choice"] = behav_choice
                                            # st.experimental_rerun()
                                    else:
                                        if st.session_state['refinements'][behav_choice][i]["submitted"] == False:
                                            st.session_state['refinements'][behav_choice][i][
                                                "choice"] = None
                                            # st.experimental_rerun()
                                st.write(st.session_state['refinements'])

                                # np.
                                # try:
                                #     if returned_choice == 'other':
                                #         new_behav = colR.text_input('input name of behavior', )
                                #     st.write(new_behav)
                                # except:
                                #     pass
                                # st.session_state['refinements'][behav_choice][i] = returned_choice
                                # user_choices[behav_choice][i] = returned_choice
                                # st.write(st.session_state['refinements'])
                                # st.write(user_choices)

    # else:
    # st.error(NO_CONFIG_HELP)
