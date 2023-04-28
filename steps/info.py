import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils.import_utils import *


def main():
    st.markdown(f" <h1 style='text-align: left; color: #FF9B24; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>Welcome to BehaveTrack</h1> "
                , unsafe_allow_html=True)
    st.write("---")
    st.markdown(f" <h1 style='text-align: left; color: #5C5C5C; font-size:18px; "
                f"font-family:Avenir; font-weight:normal'>"
                f"BehaveTrack uses user-specific B-SOiD/A-SOiD classifier to predict behavior based on pose. "
                f"Generate ethogram and summary statistics for animal recordings. "
                f"Intuitive interface for tagging and grouping files. "
                f"Enhance animal behavior studies with accurate and reliable predictions."
                f""
                f"</h1> "
                , unsafe_allow_html=True)

    bsoid_t, asoid_t = st.tabs(['B-SOiD', 'A-SOiD'])
    with bsoid_t:
        try:
            if 'bsoid_classifier' in st.session_state:
                text_ = f":orange[**RESET**] b-soid classifier in memory!"

            def clear_classifier():
                del st.session_state['bsoid_classifier']

            st.button(text_, on_click=clear_classifier)

        except:
            st.markdown(f" <h1 style='text-align: left; color: #FF9B24; font-size:18px; "
                        f"font-family:Avenir; font-weight:normal'>Upload B-SOiD _randomforest.sav</h1> "
                        , unsafe_allow_html=True)
            uploaded_rf_file = st.file_uploader('',
                                                type='sav',
                                                label_visibility='collapsed')
            try:
                if 'bsoid_classifier' not in st.session_state:
                    [_, _, _, st.session_state['bsoid_classifier'], _, _] = joblib.load(uploaded_rf_file)
            except:
                st.warning('please upload sav file')
            if 'bsoid_classifier' in st.session_state:
                st.experimental_rerun()
    with asoid_t:
        try:
            if 'asoid_classifier' in st.session_state:
                text_ = f":orange[**RESET**] a-soid classifier in memory!"

            def clear_classifier():
                del st.session_state['asoid_classifier']

            st.button(text_, on_click=clear_classifier)

        except:
            st.markdown(f" <h1 style='text-align: left; color: #FF9B24; font-size:18px; "
                        f"font-family:Avenir; font-weight:normal'>Upload A-SOiD .sav files</h1> "
                        , unsafe_allow_html=True)
            file1, file2 = st.columns(2)
            feats_targets_file = file1.file_uploader('Upload your feats_targets.sav',
                                                     accept_multiple_files=False,
                                                     type='sav',
                                                     key=f'asoid_featstargets_pkl')
            training_file = file2.file_uploader('Upload your iterX.sav',
                                                accept_multiple_files=False,
                                                type='sav',
                                                key=f'asoid_training_pkl')

            try:
                if 'asoid_classifier' not in st.session_state:
                    st.session_state['asoid_classifier'] = rescale_classifier(feats_targets_file, training_file)
            except:
                st.warning('please upload sav file')
            if 'asoid_classifier' in st.session_state:
                st.experimental_rerun()

    # try:
    #     if 'bsoid_classifier' in st.session_state:
    #         text_ = f":orange[**RESET**] b-soid classifier in memory!"
    #
    #     def clear_classifier():
    #         del st.session_state['classifier']
    #
    #     st.button(text_, on_click=clear_classifier)
    #
    # except:
    #     st.markdown(f" <h1 style='text-align: left; color: #FF9B24; font-size:18px; "
    #                 f"font-family:Avenir; font-weight:normal'>Upload B-SOiD _randomforest.sav</h1> "
    #                 , unsafe_allow_html=True)
    #     uploaded_rf_file = st.file_uploader('',
    #                                         type='sav',
    #                                         label_visibility='collapsed')
    #     try:
    #         if 'classifier' not in st.session_state:
    #             [_, _, _, st.session_state['classifier'], _, _] = joblib.load(uploaded_rf_file)
    #             # pickle.dump(st.session_state['classifier'], open('./models/demo.pkl', 'wb'))
    #     except:
    #         st.warning('please upload sav file')
    #     if 'classifier' in st.session_state:
    #         st.experimental_rerun()

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"BehaveTrack is developed by Alexander Hsu</h1> "
                    , unsafe_allow_html=True)