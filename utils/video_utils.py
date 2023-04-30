import os
import re

import cv2
import ffmpeg
from moviepy.editor import VideoFileClip

from utils.classifier_utils import *


def frame_extraction(video_file, frame_dir):
    probe = ffmpeg.probe(video_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    bit_rate = int(video_info['bit_rate'])
    avg_frame_rate = round(
        int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(
            video_info['avg_frame_rate'].rpartition('/')[2]))
    if st.button('Start frame extraction for {} frames '
                 'at {} frames per second'.format(num_frames, avg_frame_rate)):
        st.info('Extracting frames from the video... ')
        try:
            (ffmpeg.input(video_file)
             .filter('fps', fps=avg_frame_rate)
             .output(str.join('', (frame_dir, '/frame%01d.png')), video_bitrate=bit_rate,
                     s=str.join('', (str(int(width * 0.5)), 'x', str(int(height * 0.5)))),
                     sws_flags='bilinear', start_number=0)
             .run(capture_stdout=True, capture_stderr=True))
            st.info(
                'Done extracting **{}** frames from video **{}**.'.format(num_frames, video_file))
        except ffmpeg.Error as e:
            st.error('stdout:', e.stdout.decode('utf8'))
            st.error('stderr:', e.stderr.decode('utf8'))
        st.info('Done extracting {} frames from {}'.format(num_frames, video_file))


def convert_int(s):
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s):
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


def create_behavior_snippets(labels, counts,
                             framerate, output_fps,
                             frame_dir, output_path):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """

    annotation_classes = st.session_state['classifier'].classes_
    st.write(annotation_classes)
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    number_of_frames = int(framerate / 10)

    # Center coordinates
    center_coordinates = (50, 50)
    # Radius of circle
    radius = 20
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness_circle = -1

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (width - 50, height - 50)
    fontScale = 0.5
    fontColor = (0, 0, 255)
    thickness_text = 1
    lineType = 2

    for b in np.unique(labels):
        with st.spinner(f'generating videos for behavior {annotation_classes[int(b)]}'):
            idx_b = np.where(labels == b)[0]
            try:
                examples_b = np.random.choice(idx_b, counts, replace=False)
            except:
                examples_b = np.random.choice(idx_b, len(idx_b), replace=False)

            for ex, example_b in enumerate(stqdm(examples_b, desc="creating videos")):
                video_name = 'behavior_{}_example_{}.mp4'.format(annotation_classes[int(b)], int(ex))
                grp_images = []

                for f in range(number_of_frames):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                    # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)
                    grp_images.append(rgb_im)

                for f in range(number_of_frames, int(2 * number_of_frames)):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))

                    # Draw a circle with blue line borders of thickness of 2 px
                    rgb_im = cv2.circle(rgb_im, center_coordinates, radius, color, thickness_circle)

                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)

                    grp_images.append(rgb_im)

                for f in range(int(2 * number_of_frames), int(3 * number_of_frames)):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                    # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)
                    grp_images.append(rgb_im)

                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps, (width, height))
                for j, image in enumerate(grp_images):
                    video.write(image)
                cv2.destroyAllWindows()
                video.release()
                videoClip = VideoFileClip(os.path.join(output_path, video_name))
                vid_prefix = video_name.rpartition('.mp4')[0]
                gif_name = f"{vid_prefix}.gif"
                videoClip.write_gif(os.path.join(output_path, gif_name))
    return


# def create_videos(processed_input_data, rf_model, framerate,
#                   num_outliers, output_fps, annotation_classes,
#                   frame_dir, shortvid_dir):
#     if st.button("Predict labels and create example videos"):
#         st.info('Predicting labels... ')
#
#         # extract features, bin them
#         for i, data in enumerate(processed_input_data):
#             predict = frameshift_predict(processed_input_data, 1, rf_model, framerate=30)
#         create_behavior_snippets(predict, num_outliers,
#                                  framerate, output_fps, annotation_classes,
#                                  frame_dir, shortvid_dir)
#         st.balloons()



def create_videos(processed_input_data, iterX_model, framerate,
                  num_outliers, output_fps,
                  frame_dir, shortvid_dir):
    if st.button("Predict labels and create example videos"):
        st.info('Predicting labels... ')
        feats = feature_extraction(processed_input_data, 1, framerate)
        predict = bsoid_predict_numba([feats[0]], iterX_model)
        predict_arr = np.array(predict).flatten()
        create_labeled_vid(predict_arr, num_outliers,
                           framerate, output_fps,
                           frame_dir, shortvid_dir)
        st.balloons()


def convert_int(s):
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s):
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


def create_labeled_vid(labels, counts,
                       framerate, output_fps,
                       frame_dir, output_path):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    annotation_classes = np.arange(len(st.session_state['annotations']))
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    number_of_frames = int(framerate / 10)

    # Center coordinates
    center_coordinates = (50, 50)
    # Radius of circle
    radius = 20
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness_circle = -1

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (width - 50, height - 50)
    fontScale = 0.5
    fontColor = (0, 0, 255)
    thickness_text = 1
    lineType = 2

    for b in np.unique(labels):
        with st.spinner(f'generating videos for behavior {annotation_classes[int(b)]}'):
            idx_b = np.where(labels == b)[0]
            try:
                examples_b = np.random.choice(idx_b, counts, replace=False)
            except:
                examples_b = np.random.choice(idx_b, len(idx_b), replace=False)

            for ex, example_b in enumerate(stqdm(examples_b, desc="creating videos")):
                video_name = 'behavior_{}_example_{}.mp4'.format(annotation_classes[int(b)], int(ex))
                grp_images = []

                for f in range(number_of_frames):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                    # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)
                    grp_images.append(rgb_im)

                for f in range(number_of_frames, int(2 * number_of_frames)):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))

                    # Draw a circle with blue line borders of thickness of 2 px
                    rgb_im = cv2.circle(rgb_im, center_coordinates, radius, color, thickness_circle)

                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)

                    grp_images.append(rgb_im)

                for f in range(int(2 * number_of_frames), int(3 * number_of_frames)):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                    # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)
                    grp_images.append(rgb_im)

                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps,
                                        (int(width), int(height)))
                for j, image in enumerate(grp_images):
                    video.write(image)
                cv2.destroyAllWindows()
                video.release()
                videoClip = VideoFileClip(os.path.join(output_path, video_name))
                vid_prefix = video_name.rpartition('.mp4')[0]
                gif_name = f"{vid_prefix}.gif"
                videoClip.write_gif(os.path.join(output_path, gif_name))
    return