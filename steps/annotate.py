import base64
import glob
import io
from pathlib import Path
import json
from utils.video_utils import *


def prompt_setup(working_dir=Path.home(), prefix='Desktop/behave_track'):
    left_col, right_col = st.columns(2)
    left_expand = left_col.expander('Select a video file:', expanded=True)
    right_expand = right_col.expander('Select the corresponding pose file:', expanded=True)
    temporary_location = None
    framerate = None
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
        if os.path.exists(new_videos.name):
            temporary_location = f'{new_videos.name}'
        else:
            g = io.BytesIO(new_videos.read())  # BytesIO Object
            temporary_location = f'{new_videos.name}'
            with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            out.close()
        probe = ffmpeg.probe(temporary_location)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        framerate = round(
            int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(
                video_info['avg_frame_rate'].rpartition('/')[2]))
    except:
        st.warning('please upload both video and pose file')

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
        num_outliers = col1_exp.number_input('Number of examples to define your behavior',
                                             min_value=3, max_value=None, value=5)
        output_fps = col1_exp.number_input('Video playback fps',
                                           min_value=2, max_value=None, value=10)
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
        if 'annotations' not in st.session_state:
            st.session_state['annotations'] = {key: {'name':None}
                                               for key in range(st.session_state['classifier'].n_classes_)}
    except:
        pass
    return temporary_location, framerate, new_pose_list, num_outliers, output_fps, frame_dir, shortvid_dir


def main():
    try:
        if 'bsoid_classifier' in st.session_state:
            st.session_state['classifier'] = st.session_state['bsoid_classifier']
            st.write('B-SOiD classifier loaded!')
        elif 'asoid_classifier' in st.session_state:
            st.session_state['classifier'] = st.session_state['asoid_classifier']
            st.write('A-SOiD classifier loaded!')
    except:
        st.warning('please return to home to upload sav files')
    [temporary_location, framerate, new_pose_list, num_outliers, output_fps, frame_dir, shortvid_dir] = \
        prompt_setup()

    try:
        infilename = str.join('',
                               (os.path.join(Path.home(),
                                             'Desktop/behave_track'),
                                '/behavior_names.npy'))
        prev_annotation = np.load(infilename, allow_pickle=True).item()
        if 'annotations' not in st.session_state:
            st.session_state['annotations'] = prev_annotation
        # st.write(st.session_state['annotations'])
    except:
        pass

    # try:
    if temporary_location is not None and len(new_pose_list) > 0:
        if os.path.exists(frame_dir):
            framedir_ = os.listdir(frame_dir)
            if len(framedir_) < 2:
                frame_extraction(video_file=temporary_location, frame_dir=frame_dir)
            else:
                if os.path.exists(shortvid_dir):
                    viddir_ = os.listdir(shortvid_dir)
                    if len(viddir_) < 2:
                        create_videos(new_pose_list, st.session_state['classifier'],
                                      framerate,
                                      num_outliers, output_fps,
                                      frame_dir=frame_dir, shortvid_dir=shortvid_dir)

                    else:
                        col_option, col_msg = st.columns(2)
                        if col_option.checkbox('Redo? Uncheck after check to prevent from auto-clearing',
                                               False,
                                               key='vr'):
                            try:
                                for file_name in glob.glob(shortvid_dir + "/*"):
                                    os.remove(file_name)
                            except:
                                pass

                        behav_choice = st.selectbox("Select the behavior: ", st.session_state['annotations'],
                                                    index=int(0),
                                                    key="behavior_choice")
                        alltabs = st.tabs([f'{i}' for i in range(num_outliers)])
                        for i, tab_ in enumerate(alltabs):
                            with tab_:
                                colL, colR = st.columns([4, 1.5])
                                colL_exp = colL.expander('example video', expanded=True)
                                colR_exp = colR.expander('input your definition', expanded=True)
                                file_ = open(
                                    os.path.join(shortvid_dir, f'behavior_{behav_choice}_example_{i}.gif'),
                                    "rb")
                                contents = file_.read()
                                data_url = base64.b64encode(contents).decode("utf-8")
                                file_.close()
                                gif_width = colL_exp.slider('width',
                                                            min_value=200, max_value=1000, value=600,
                                                            key=f'slider_{behav_choice}_{i}')
                                colL_exp.markdown(
                                    f"<img src='data:image/gif;base64,{data_url}' width='{gif_width}px'>",
                                    unsafe_allow_html=True,
                                )

                                with colR_exp:
                                    if st.session_state['annotations'][behav_choice]["name"] is None:
                                        returned_def = st.text_input('what is this behavior?',
                                                                     key=f'input_box_{behav_choice}_{i}')
                                        def save_user_input(returned_def):
                                            st.session_state['annotations'][behav_choice]["name"] = returned_def
                                            st.experimental_rerun()

                                        if st.button("Submit", key=f'submit_button_{behav_choice}_{i}'):
                                            save_user_input(returned_def)

                                    else:
                                        st.markdown(
                                            f" <h1 style='text-align: left; color: #FF9B24; font-size:16px; "
                                            f"font-family:Avenir; font-weight:normal'> Behavior name: "
                                            f" {st.session_state['annotations'][behav_choice]['name']} </h1> "
                                            , unsafe_allow_html=True)
                                        def clear_user_input():
                                            st.session_state['annotations'][behav_choice]["name"] = None
                                            st.experimental_rerun()
                                        if st.button("Clear", key=f'clear_button_{behav_choice}_{i}'):
                                            clear_user_input()
                                outfilename = str.join('',
                                                       (os.path.join(Path.home(),
                                                                     'Desktop/behave_track'),
                                                        '/behavior_names.npy'))
                                np.save(outfilename, st.session_state['annotations'])
                                # with open(outfilename, 'w') as f:
                                #     for key, value in st.session_state['annotations'].items():
                                #         f.write('%s:%s\n' % (key, value))

    # except:
    #     pass
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"BehaveTrack is developed by Alexander Hsu</h1> "
                    , unsafe_allow_html=True)
