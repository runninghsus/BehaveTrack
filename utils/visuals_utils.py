import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter

from utils.classifier_utils import *
from utils.download_utils import *


def ethogram_plot(condition, new_predictions, behavior_names, behavior_colors, length_):
    colL, colR = st.columns(2)
    if len(new_predictions) == 1:
        colL.markdown(':orange[1] file only')
        f_select = 0
    else:
        f_select = colL.slider('select file to generate ethogram',
                               min_value=1, max_value=len(new_predictions), value=1,
                               key=f'ethogram_slider_{condition}')
    file_idx = f_select - 1
    prefill_array = np.zeros((len(new_predictions[file_idx]),
                              len(st.session_state['classifier'].classes_)))
    default_colors_wht = ['w']
    default_colors_wht.extend(behavior_colors)
    cmap_ = ListedColormap(default_colors_wht)

    count = 0
    for b in np.unique(st.session_state['classifier'].classes_):
        idx_b = np.where(new_predictions[file_idx] == b)[0]
        prefill_array[idx_b, count] = b + 1
        count += 1
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    seed_num = colR.number_input('seed for segment',
                                 min_value=0, max_value=None, value=42,
                                 key=f'cond{condition}_seed')
    np.random.seed(seed_num)
    behaviors_with_names = behavior_names
    if colL.checkbox('use randomized time',
                     value=True,
                     key=f'cond{condition}_ckbx'):
        rand_start = np.random.choice(prefill_array.shape[0] - length_, 1, replace=False)
        ax.imshow(prefill_array[int(rand_start):int(rand_start + length_), :].T, cmap=cmap_)
        ax.set_xticks(np.arange(0, length_, int(length_ / 5)))
        ax.set_xticklabels(np.arange(int(rand_start), int(rand_start + length_), int(length_ / 5)) / 10)
        ax.set_yticks(np.arange(len(behaviors_with_names)))
        ax.set_yticklabels(behaviors_with_names)
        ax.set_xlabel('seconds')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        rand_start = 0
        ax.imshow(prefill_array[rand_start:rand_start + length_, :].T, cmap=cmap_)
        ax.set_xticks(np.arange(rand_start, length_, int(length_ / 5)))
        ax.set_xticklabels(np.arange(0, length_, int(length_ / 5)) / 10)
        ax.set_yticks(np.arange(len(behaviors_with_names)))
        ax.set_yticklabels(behaviors_with_names)
        ax.set_xlabel('seconds')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, prefill_array, rand_start


def ethogram_predict(placeholder, condition, behavior_colors, length_):
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))

    with placeholder:
        etho_placeholder = st.empty()
        fig, prefill_array, rand_start = ethogram_plot(condition, predict, behavior_classes,
                                                       list(behavior_colors.values()), length_)
        etho_placeholder.pyplot(fig)


def condition_etho_plot():
    behavior_classes = st.session_state['classifier'].classes_
    length_container = st.container()
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    length_ = length_container.slider('number of frames',
                                      min_value=25, max_value=250,
                                      value=75,
                                      key=f'length_slider')
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        ethogram_predict(left_expander,
                         list(st.session_state['features'].keys())[count],
                         behavior_colors, length_)
        predict_csv = csv_predict(
            list(st.session_state['features'].keys())[count],
        )

        left_expander.download_button(
            label="Download data as CSV",
            data=predict_csv,
            file_name=f"predictions_{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                ethogram_predict(right_expander,
                                 list(st.session_state['features'].keys())[count],
                                 behavior_colors, length_)
                predict_csv = csv_predict(
                    list(st.session_state['features'].keys())[count],
                )

                right_expander.download_button(
                    label="Download data as CSV",
                    data=predict_csv,
                    file_name=f"predictions_{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            ethogram_predict(right_expander,
                             list(st.session_state['features'].keys())[count],
                             behavior_colors, length_)
            predict_csv = csv_predict(
                list(st.session_state['features'].keys())[count],
            )

            right_expander.download_button(
                label="Download data as CSV",
                data=predict_csv,
                file_name=f"predictions_{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def pie_predict(placeholder, condition, behavior_colors):
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    predict = []
    # TODO: find a color workaround if a class is missing
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    predict_dict = {'condition': np.repeat(condition, len(np.hstack(predict))),
                    'behavior': np.hstack(predict)}
    df_raw = pd.DataFrame(data=predict_dict)
    labels = df_raw['behavior'].value_counts(sort=False).index
    values = df_raw['behavior'].value_counts(sort=False).values
    # summary dataframe
    df = pd.DataFrame()
    behavior_labels = []
    for l in labels:
        behavior_labels.append(behavior_classes[int(l)])
    df["values"] = values
    df['labels'] = behavior_labels
    df["colors"] = df["labels"].apply(lambda x:
                                      behavior_colors.get(x))  # to connect Column value to Color in Dict
    with placeholder:
        fig = go.Figure(data=[go.Pie(labels=df["labels"], values=df["values"], hole=.4)])
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='value',
                          textfont_size=16,
                          marker=dict(colors=df["colors"],
                                      line=dict(color='#000000', width=1)))
        st.plotly_chart(fig, use_container_width=True)


def condition_pie_plot():
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        pie_predict(left_expander,
                    list(st.session_state['features'].keys())[count],
                    behavior_colors)
        totaldur_csv = duration_pie_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=totaldur_csv,
            file_name=f"total_durations_{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                pie_predict(right_expander,
                            list(st.session_state['features'].keys())[count],
                            behavior_colors)
                totaldur_csv = duration_pie_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=totaldur_csv,
                    file_name=f"total_durations_{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            pie_predict(right_expander,
                        list(st.session_state['features'].keys())[count],
                        behavior_colors)
            totaldur_csv = duration_pie_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=totaldur_csv,
                file_name=f"total_durations_{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def location_predict(placeholder, condition, behavior_colors):
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    # names = [f'behavior {int(key)}' for key in behavior_classes]
    pose = st.session_state['pose'][condition]
    bp_select = placeholder.radio('', st.session_state['bodypart_names'], horizontal=True, label_visibility='collapsed',
                                  key=f'bodypart_radio_{condition}')
    behav_selects = placeholder.multiselect('select behavior to visualize location', behavior_classes,
                                            default=behavior_classes[0],
                                            key=f'behavior_multiselect_{condition}')
    bodypart_idx = st.session_state['bodypart_names'].index(bp_select) * 2
    predict = []
    for f in range(len(st.session_state['features'][condition])):

        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))

    colL, colR = placeholder.columns(2)
    if len(predict) == 1:
        colL.markdown(':orange[1] file only')
        f_select = 0
    else:
        f_select = colL.slider('select file to generate ethogram',
                               min_value=1, max_value=len(predict), value=1,
                               key=f'ethogram_slider_{condition}')
    file_chosen = f_select - 1
    fig, ax = plt.subplots(1, 1)
    for b, behavior_name in enumerate(behavior_classes):
        idx_b = np.where(predict[file_chosen] == b)[0]
        if behavior_name in behav_selects:
            ax.scatter(pose[file_chosen][idx_b, bodypart_idx],
                       pose[file_chosen][idx_b, bodypart_idx + 1],
                       c=behavior_colors[behavior_name])
        else:
            ax.scatter(pose[file_chosen][idx_b, bodypart_idx],
                       pose[file_chosen][idx_b, bodypart_idx + 1],
                       c=behavior_colors[behavior_name], alpha=0.0)
    # minx, miny = np.min(pose[file_chosen][:, bodypart_idx]), np.min(pose[file_chosen][:, bodypart_idx+1])
    # maxx, maxy = np.max(pose[file_chosen][:, bodypart_idx]), np.max(pose[file_chosen][:, bodypart_idx+1])
    plt.axis('off')
    plt.axis('equal')
    placeholder.pyplot(fig)

    # predict_dict = {'condition': np.repeat(condition, len(np.hstack(predict))),
    #                 'behavior': np.hstack(predict)}
    # df_raw = pd.DataFrame(data=predict_dict)
    # labels = df_raw['behavior'].value_counts(sort=False).index
    # values = df_raw['behavior'].value_counts(sort=False).values
    # names = [f'behavior {int(key)}' for key in behavior_classes]
    # # summary dataframe
    # df = pd.DataFrame()
    # # do i need this?
    # behavior_labels = []
    # for l in labels:
    #     behavior_labels.append(behavior_classes[int(l)])
    # df["values"] = values
    # df['labels'] = behavior_labels
    # df["colors"] = df["labels"].apply(lambda x:
    #                                   behavior_colors.get(x))  # to connect Column value to Color in Dict
    # with placeholder:
    #     fig = go.Figure(data=[go.Pie(labels=[names[int(i)] for i in df["labels"]], values=df["values"], hole=.4)])
    #     fig.update_traces(hoverinfo='label+percent',
    #                       textinfo='value',
    #                       textfont_size=16,
    #                       marker=dict(colors=df["colors"],
    #                                   line=dict(color='#000000', width=1)))
    #     st.plotly_chart(fig, use_container_width=True)


def condition_location_plot():
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        location_predict(left_expander,
                         list(st.session_state['features'].keys())[count],
                         behavior_colors)
        totaldur_csv = duration_pie_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=totaldur_csv,
            file_name=f"location_{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                location_predict(right_expander,
                                 list(st.session_state['features'].keys())[count],
                                 behavior_colors)
                totaldur_csv = duration_pie_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=totaldur_csv,
                    file_name=f"location_{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            location_predict(right_expander,
                             list(st.session_state['features'].keys())[count],
                             behavior_colors)
            totaldur_csv = duration_pie_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=totaldur_csv,
                file_name=f"location_{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def bar_predict(placeholder, condition, behavior_colors):
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        bout_counts = []
        for file_idx in range(len(predict)):
            bout_counts.append(get_num_bouts(predict[file_idx], behavior_classes))
        bout_mean = np.mean(bout_counts, axis=0)
        bout_std = np.std(bout_counts, axis=0)
        # names = [f'behavior {int(key)}' for key in behavior_classes]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=behavior_classes, y=bout_mean,
            # name=names,
            error_y=dict(type='data', array=bout_std),
            width=0.5,
            marker_color=pd.Series(behavior_colors),
            marker_line=dict(width=1.2, color='black'))
        )
        y_max = np.max(bout_mean + bout_std)
        max_counts = st.slider('behavioral instance counts y limit',
                               min_value=0,
                               max_value=int(y_max) * 2,
                               value=int(y_max),
                               key=f'max_counts_slider_{condition}')
        fig.update_layout(yaxis=dict(title=f"counts (mean+-sd) across "
                                           f"{len(st.session_state['features'][condition])} files"),
                          yaxis_range=[0, max_counts])
        st.plotly_chart(fig, use_container_width=True)


def condition_bar_plot():
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        bar_predict(left_expander,
                    list(st.session_state['features'].keys())[count],
                    behavior_colors)
        bar_csv = bout_bar_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=bar_csv,
            file_name=f"bouts_{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                bar_predict(right_expander,
                            list(st.session_state['features'].keys())[count],
                            behavior_colors)
                bar_csv = bout_bar_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=bar_csv,
                    file_name=f"bouts_{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            bar_predict(right_expander,
                        list(st.session_state['features'].keys())[count],
                        behavior_colors)
            bar_csv = bout_bar_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=bar_csv,
                file_name=f"bouts_{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def ridge_predict(placeholder, condition, behavior_colors):
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        duration_ = []
        for file_idx in range(len(predict)):
            duration_.append(get_duration_bouts(predict[file_idx], behavior_classes, framerate=10))
        colors = [mcolors.to_hex(i) for i in list(behavior_colors.values())]
        for file_chosen in range(len(duration_)):
            if file_chosen == 0:
                duration_matrix = boolean_indexing(duration_[file_chosen])
            else:
                duration_matrix = np.hstack((duration_matrix, boolean_indexing(duration_[file_chosen])))
        max_dur = st.slider('max duration',
                            min_value=0,
                            max_value=int(
                                np.nanpercentile(np.array(duration_matrix),
                                                 100)),
                            value=int(np.nanpercentile(np.array(duration_matrix),
                                                       99)),
                            key=f'maxdur_slider_{condition}')
        fig = go.Figure()
        # names = [f'behavior {int(key)}' for key in behavior_classes]
        for data_line, color, name in zip(duration_matrix, colors, behavior_classes):
            fig.add_trace(go.Violin(x=data_line,
                                    line_color=color,
                                    name=name))
        fig.update_traces(
            orientation='h', side='positive', width=3, points=False,
        )
        fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False,
                          xaxis_range=[0, max_dur], xaxis=dict(title='seconds'))
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        for data_line, color, name in zip(duration_matrix, colors, behavior_classes):
            fig.add_trace(go.Box(y=data_line,
                                 jitter=0.5,
                                 whiskerwidth=0.3,
                                 fillcolor=color,
                                 marker_size=2,
                                 line_width=1.2,
                                 line_color='#000000',
                                 name=name))
        fig.update_layout(yaxis=dict(title='bout duration (seconds)'),
                          yaxis_range=[0, max_dur],
                          )
        st.plotly_chart(fig, use_container_width=True)


def condition_ridge_plot():
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        ridge_predict(left_expander,
                      list(st.session_state['features'].keys())[count],
                      behavior_colors)
        ridge_csv = duration_ridge_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=ridge_csv,
            file_name=f"bout_durations_{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                ridge_predict(right_expander,
                              list(st.session_state['features'].keys())[count],
                              behavior_colors)
                ridge_csv = duration_ridge_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=ridge_csv,
                    file_name=f"bout_durations_{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            ridge_predict(right_expander,
                          list(st.session_state['features'].keys())[count],
                          behavior_colors)
            ridge_csv = duration_ridge_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=ridge_csv,
                file_name=f"bout_durations_{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def transmat_predict(placeholder, condition, heatmap_color_scheme):
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    # names = [f'behavior {int(key)}' for key in behavior_classes]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        transitions_ = []
        for file_idx in range(len(predict)):
            count_tm, prob_tm = get_transitions(predict[file_idx], behavior_classes)
            transitions_.append(prob_tm)
        mean_transitions = np.mean(transitions_, axis=0)
        fig = px.imshow(mean_transitions,
                        color_continuous_scale=heatmap_color_scheme,
                        aspect='equal'
                        )
        fig.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=np.arange(len(behavior_classes)),
                ticktext=behavior_classes),
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(len(behavior_classes)),
                ticktext=behavior_classes)
        )
        st.plotly_chart(fig, use_container_width=True)


def directedgraph_predict(placeholder, condition, heatmap_color_scheme):
    # behavior_classes = st.session_state['classifier'].classes_
    # names = [f'behavior {int(key)}' for key in behavior_classes]
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        transitions_count = []
        transitions_prob = []
        for file_idx in range(len(predict)):
            count_tm, prob_tm = get_transitions(predict[file_idx], behavior_classes)
            transitions_count.append(count_tm)
            transitions_prob.append(prob_tm)
        transition_count_mean = np.nanmean(transitions_count, axis=0)
        transitions_prob_mean = np.nanmean(transitions_prob, axis=0)
        diag = [transition_count_mean[i][i] for i in range(len(transition_count_mean))]
        diag_p = np.array(diag) / np.array(diag).max()
        # keep diag to provide information about relative behavioral duration
        # scale it by 50, works well, and save it as a global variable
        node_sizes = [50 * i for i in diag_p]
        ## transition matrix from 2d array into numpy matrix for networkx
        transition_prob_raw = np.matrix(transitions_prob_mean)
        # replace diagonal with 0
        np.fill_diagonal(transition_prob_raw, 0)
        transition_prob_norm = transition_prob_raw / transition_prob_raw.sum(axis=1)
        nan_indices = np.isnan(transition_prob_norm)
        transition_prob_norm[nan_indices] = 0

        fig = plt.figure(figsize=(8, 8))
        # particular networkx graph
        graph = nx.from_numpy_array(transition_prob_norm, create_using=nx.MultiDiGraph())
        # set node position with seed 0 for reproducibility
        node_position = nx.layout.spring_layout(graph, seed=0)
        # edge colors is equivalent to the weight
        edge_colors = [graph[u][v][0].get('weight') for u, v in graph.edges()]
        # TODO: try to find a way to fix vmin vmax in directed graph
        # c_max = np.max(edge_colors)
        # max_c = st.slider('color axis limit',
        #                   min_value=0.0,
        #                   max_value=1.0,
        #                   value=np.float(c_max),
        #                   key=f'max_color_slider_{condition}')

        # node is dependent on the self transitions, which is defined in compute dynamics above, use blue colormap
        nodes = nx.draw_networkx_nodes(graph, node_position, node_size=node_sizes,
                                       node_color='blue')

        # edges are drawn as arrows with blue colormap, size 8 with width 1.5
        edges = nx.draw_networkx_edges(graph, node_position, node_size=node_sizes, arrowstyle='->',
                                       arrowsize=8, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=1.5)
        # label position is 0.005 to the right of the node
        label_pos = [node_position[i] + 0.005 for i in range(len(node_position))]
        # draw labels with font size 10
        labels_dict = {}
        for i, label in enumerate(behavior_classes):
            labels_dict[i] = label
        nx.draw_networkx_labels(graph, label_pos, labels_dict, font_size=10)
        # generate colorbar from the edge colors
        pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
        # pc.set_clim([0, max_c])
        pc.set_array(edge_colors)
        plt.colorbar(pc, shrink=0.5, location='bottom')
        ax = plt.gca()
        ax.set_axis_off()
        st.pyplot(fig, use_container_width=True)


def condition_transmat_plot():
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    named_colorscales = px.colors.named_colorscales()
    col1, col2 = option_expander.columns([3, 1])
    heatmap_color_scheme = col1.selectbox(f'select colormap for heatmap',
                                          named_colorscales,
                                          index=named_colorscales.index('agsunset'),
                                          key='color_scheme')
    col2.write('')
    col2.write('')
    if col2.checkbox('reverse?'):
        heatmap_color_scheme = str.join('', (heatmap_color_scheme, '_r'))
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        transmat_predict(left_expander,
                         list(st.session_state['features'].keys())[count],
                         heatmap_color_scheme)
        directedgraph_predict(left_expander,
                              list(st.session_state['features'].keys())[count],
                              heatmap_color_scheme)
        transition_csv = transmat_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=transition_csv,
            file_name=f"transitions_{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                transmat_predict(right_expander,
                                 list(st.session_state['features'].keys())[count],
                                 heatmap_color_scheme)
                directedgraph_predict(right_expander,
                                      list(st.session_state['features'].keys())[count],
                                      heatmap_color_scheme)
                transition_csv = transmat_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=transition_csv,
                    file_name=f"transitions_{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            transmat_predict(right_expander,
                             list(st.session_state['features'].keys())[count],
                             heatmap_color_scheme)
            directedgraph_predict(right_expander,
                                  list(st.session_state['features'].keys())[count],
                                  heatmap_color_scheme)
            transition_csv = transmat_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=transition_csv,
                file_name=f"transitions_{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def kinematix_predict(placeholder, condition, behavior_colors):
    behavior_classes = st.session_state['classifier'].classes_
    names = [f'behavior {int(key)}' for key in behavior_classes]
    pose = st.session_state['pose'][condition]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        dist_tab, dur_tab, speed_tab = st.tabs(['stride examples', 'stride length/speed', 'limb stance'])
        option_container = st.container()
        plot_container = st.container()
        def_bp_selects = ['right-forepaw', 'right-hindpaw']
        # if st.checkbox(f"use default body part: {', '.join(def_bp_selects)}",
        #                key=f'default_bp_chkbx{condition}', value=True):
        #     bp_selects = def_bp_selects
        # else:
        # st.write(st.session_state['bodypart_names'].index("right-forepaw"))
        bp_selects = st.selectbox('select body part',
                                  st.session_state['bodypart_names'],
                                  index=st.session_state['bodypart_names'].index("right-forepaw"),
                                  key=f'bodypart_selectbox_{condition}')
        bout_disp_bps = []
        bout_duration_bps = []
        bout_avg_speed_bps = []
        for bp_select in [bp_selects]:
            bodypart = st.session_state['bodypart_names'].index(bp_select)
            bout_disp_all = []
            bout_duration_all = []
            bout_avg_speed_all = []
            for file_chosen in range(len(predict)):
                behavior, behavioral_start_time, behavior_duration, bout_disp, bout_duration, bout_avg_speed = \
                    get_avg_kinematics(predict[file_chosen], pose[file_chosen], bodypart, framerate=10)
                bout_disp_all.append(bout_disp)
                bout_duration_all.append(bout_duration)
                bout_avg_speed_all.append(bout_avg_speed)
            bout_disp_bps.append(bout_disp_all)
            bout_duration_bps.append(bout_duration_all)
            bout_avg_speed_bps.append(bout_avg_speed_all)

        behavioral_sums = {key: [] for key in names}
        behavioral_dur = {key: [] for key in names}
        behavioral_speed = {key: [] for key in names}
        # sum over bouts
        # bp, file, behav, instance
        with dist_tab:
            # bp, file, behav, instance
            slider_, checkbox_ = option_container.columns([3, 1])
            y_val = []
            checkbox_.write('')
            checkbox_.write('')
            if checkbox_.checkbox('smooth',
                                  key=f'smth_checkbox_{condition}'):
                for inst in range(len(bout_disp_bps[0][0][2])):
                    try:
                        y_val.append(savgol_filter(np.hstack(bout_disp_bps[0][0][2][inst]), 3, 1))
                    except:
                        y_val.append(np.hstack(bout_disp_bps[0][0][2][inst]))
            else:
                for inst in range(len(bout_disp_bps[0][0][2])):
                    y_val.append(np.hstack(bout_disp_bps[0][0][2][inst]))
            traject_dict = {'locomotor bout #': np.hstack([inst * np.ones(len(bout_disp_bps[0][0][2][inst]))
                                                           for inst in range(len(bout_disp_bps[0][0][2]))]),
                            'x': np.hstack([np.arange(len(bout_disp_bps[0][0][2][inst]))
                                            for inst in range(len(bout_disp_bps[0][0][2]))]),
                            'y': np.hstack(y_val)}

            num_choose = slider_.slider('choose number of examples:',
                                        min_value=1, max_value=len(np.unique(traject_dict['locomotor bout #'])),
                                        value=int(len(np.unique(traject_dict['locomotor bout #'])) / 2),
                                        key=f'numex_slider_{condition}')

            sampled = np.random.choice(np.unique(traject_dict['locomotor bout #']), num_choose, replace=False)
            traj_df = pd.DataFrame(data=traject_dict)
            traj_df = traj_df[traj_df['locomotor bout #'].isin(sampled)]
            sns.set_theme(rc={"axes.facecolor": (0, 0, 0, 0), 'figure.facecolor': '#ffffff', 'axes.grid': False})
            g = sns.FacetGrid(traj_df, row='locomotor bout #', hue='locomotor bout #',
                              palette=sns.color_palette("husl", num_choose),
                              aspect=25, height=0.4)
            # Draw the densities in a few steps
            g.map(plt.plot, 'x', 'y', clip_on=False, alpha=1, linewidth=2).add_legend()
            g.map(plt.fill_between, 'x', 'y', color='#ffffff', alpha=0.7)
            # g.map(sns.lineplot, 'x', 'y', clip_on=False, alpha=1, color='#000000', linewidth=2)
            # Set the subplots to overlap
            g.fig.subplots_adjust(hspace=-0.9)
            g.set_titles("")
            g.set(yticks=[], xticks=[], ylabel="", xlabel="")
            g.despine(bottom=True, left=True)
            plot_container.pyplot(g, use_container_width=True)

            # st.pyplot(g)

            # for b, behav in enumerate(behavioral_sums.keys()):
            #     for f in range(len(bout_disp_bps[0])):
            #         if f == 0:
            #             behavioral_sums[behav] = np.hstack(
            #                 [np.hstack(
            #                     [np.sum(bout_disp_bps[bp][f][b][inst])
            #                      for inst in range(len(bout_disp_bps[0][f][b]))])
            #                     for bp in range(len(bout_disp_bps))])
            #
            #         else:
            #             behavioral_sums[behav] = np.hstack((behavioral_sums[behav],
            #                                                 np.hstack(
            #                                                     [np.hstack(
            #                                                         [np.sum(bout_disp_bps[bp][f][b][inst])
            #                                                          for inst in range(len(bout_disp_bps[0][f][b]))])
            #                                                         for bp in range(len(bout_disp_bps))])))
            # fig = go.Figure()
            # y_max = 0
            # for b, behav in enumerate(behavioral_sums.keys()):
            #     fig.add_trace(go.Box(
            #         y=behavioral_sums[behav]
            #         [(behavioral_sums[behav] < np.percentile(behavioral_sums[behav], 95)) &
            #          (behavioral_sums[behav] > np.percentile(behavioral_sums[behav], 5))],
            #         name=behav,
            #         line_color=behavior_colors[b],
            #         boxpoints=False,
            #     ))
            #     try:
            #         if np.max(behavioral_sums[behav]
            #                   [(behavioral_sums[behav] < np.percentile(behavioral_sums[behav], 95)) &
            #                    (behavioral_sums[behav] > np.percentile(behavioral_sums[behav], 5))]) > y_max:
            #             y_max = np.max(behavioral_sums[behav]
            #                            [(behavioral_sums[behav] < np.percentile(behavioral_sums[behav], 95)) &
            #                             (behavioral_sums[behav] > np.percentile(behavioral_sums[behav], 5))])
            #     except:
            #         pass
            # max_dist_y = st.slider('pose trajectory distance y limit',
            #                        min_value=0,
            #                        max_value=int(y_max) * 2,
            #                        value=int(y_max),
            #                        key=f'max_dist_slider_{condition}')
            # fig.update_layout(yaxis=dict(title='bout distance traveled (Δpixels)'), yaxis_range=[0, max_dist_y])
            # st.plotly_chart(fig, use_container_width=True)

        with dur_tab:
            for b, behav in enumerate(behavioral_dur.keys()):
                for f in range(len(bout_duration_bps[0])):
                    if f == 0:
                        behavioral_dur[behav] = np.hstack(
                            [bout_duration_bps[bp][f][b]
                             for bp in range(len(bout_duration_bps))])

                    else:
                        behavioral_dur[behav] = np.hstack((behavioral_dur[behav],
                                                           np.hstack(
                                                               [bout_duration_bps[bp][f][b]
                                                                for bp in range(len(bout_duration_bps))])))
            fig = go.Figure()
            y_max = 0
            for b, behav in enumerate(behavioral_dur.keys()):
                fig.add_trace(go.Box(
                    y=behavioral_dur[behav]
                    [(behavioral_dur[behav] < np.percentile(behavioral_dur[behav], 95)) &
                     (behavioral_dur[behav] > np.percentile(behavioral_dur[behav], 5))],
                    name=behav,
                    line_color=behavior_colors[b],
                    boxpoints=False,
                ))
                try:
                    if np.max(behavioral_dur[behav]
                              [(behavioral_dur[behav] < np.percentile(behavioral_dur[behav], 95)) &
                               (behavioral_dur[behav] > np.percentile(behavioral_dur[behav], 5))]) > y_max:
                        y_max = np.max(behavioral_dur[behav]
                                       [(behavioral_dur[behav] < np.percentile(behavioral_dur[behav], 95)) &
                                        (behavioral_dur[behav] > np.percentile(behavioral_dur[behav], 5))])
                except:
                    pass
            max_dur_y = st.slider('bout duration y limit',
                                  min_value=0,
                                  max_value=int(y_max) * 2,
                                  value=int(y_max),
                                  key=f'max_dur_slider_{condition}')
            fig.update_layout(yaxis=dict(title='bout duration (seconds)'), yaxis_range=[0, max_dur_y])
            st.plotly_chart(fig, use_container_width=True)
        with speed_tab:
            for b, behav in enumerate(behavioral_speed.keys()):
                for f in range(len(bout_avg_speed_bps[0])):
                    if f == 0:
                        behavioral_speed[behav] = np.hstack(
                            [bout_avg_speed_bps[bp][f][b]
                             for bp in range(len(bout_avg_speed_bps))])

                    else:
                        behavioral_speed[behav] = np.hstack((behavioral_speed[behav],
                                                             np.hstack(
                                                                 [bout_avg_speed_bps[bp][f][b]
                                                                  for bp in range(len(bout_avg_speed_bps))])))
            fig = go.Figure()
            y_max = 0
            for b, behav in enumerate(behavioral_speed.keys()):
                fig.add_trace(go.Box(
                    y=behavioral_speed[behav]
                    [(behavioral_speed[behav] < np.percentile(behavioral_speed[behav], 95)) &
                     (behavioral_speed[behav] > np.percentile(behavioral_speed[behav], 5))],
                    name=behav,
                    line_color=behavior_colors[b],
                    boxpoints=False,
                ))
                try:
                    if np.max(behavioral_speed[behav]
                              [(behavioral_speed[behav] < np.percentile(behavioral_speed[behav], 95)) &
                               (behavioral_speed[behav] > np.percentile(behavioral_speed[behav], 5))]) > y_max:
                        y_max = np.max(behavioral_speed[behav]
                                       [(behavioral_speed[behav] < np.percentile(behavioral_speed[behav], 95)) &
                                        (behavioral_speed[behav] > np.percentile(behavioral_speed[behav], 5))])
                except:
                    pass
            max_speed_y = st.slider('average speed y limit',
                                    min_value=0,
                                    max_value=int(y_max) * 2,
                                    value=int(y_max),
                                    key=f'max_speed_slider_{condition}')
            fig.update_layout(yaxis=dict(title='bout movement speed (Δpixels/second)'), yaxis_range=[0, max_speed_y])
            st.plotly_chart(fig, use_container_width=True)


def condition_kinematix_plot():
    behavior_classes = st.session_state['classifier'].classes_
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        kinematix_predict(left_expander,
                          list(st.session_state['features'].keys())[count],
                          behavior_colors)
        # ridge_csv = kinematics_csv(
        #     list(st.session_state['features'].keys())[count],
        # )
        # left_expander.download_button(
        #     label="Download data as CSV",
        #     data=ridge_csv,
        #     file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
        #     mime='text/csv',
        #     key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        # )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                kinematix_predict(right_expander,
                                  list(st.session_state['features'].keys())[count],
                                  behavior_colors)
                # ridge_csv = kinematics_csv(
                #     list(st.session_state['features'].keys())[count],
                # )
                # right_expander.download_button(
                #     label="Download data as CSV",
                #     data=ridge_csv,
                #     file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                #     mime='text/csv',
                #     key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                # )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            kinematix_predict(right_expander,
                              list(st.session_state['features'].keys())[count],
                              behavior_colors)
            # ridge_csv = kinematics_csv(
            #     list(st.session_state['features'].keys())[count],
            # )
            # right_expander.download_button(
            #     label="Download data as CSV",
            #     data=ridge_csv,
            #     file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
            #     mime='text/csv',
            #     key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            # )
            count += 1
