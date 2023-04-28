import streamlit as st
from steps import info, annotate, analysis
from streamlit_option_menu import option_menu
# import streamlit_authenticator as stauth
# import yaml
# from yaml.loader import SafeLoader
from PIL import Image
from pathlib import Path
import base64


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, width=500):
    img_html = f"<img src='data:image/png;base64,{img_to_bytes(img_path)}'  width='{width}px', class='img-fluid'>"
    return img_html


icon = './images/icon.png'
banner = './images/banner.png'

icon_img = Image.open(icon)
st.set_page_config(layout="wide",
                   page_title='BehaveTrack',
                   page_icon=icon_img)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
        min-width: 250px;
        max-width: 250px;   
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
#
# with open('./config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)
#
#
# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

logo_placeholder = st.empty()
st.write('')
st.write('')
_, mid, _ = st.columns([1, 4, 1])
# with mid:
#     name, authentication_status, username = authenticator.login('Login', 'main')
# if not authentication_status:
#     logo_placeholder.markdown("<p style='text-align: center; color: grey; '>" + img_to_html(banner, width=400) + "</p>",
#                               unsafe_allow_html=True)
#     bottom_cont = st.container()
#     _, bottom_mid, _ = bottom_cont.columns([1, 4, 1])
#     with bottom_mid:
#         # st.markdown("""---""")
#         st.markdown(f" <h1 style='text-align: center; color: gray; font-size:16px; "
#                     f"font-family:Avenir; font-weight:normal'>"
#                     f"LUPE B-SOiD is developed by Alexander Hsu and Justin James</h1> "
#                     , unsafe_allow_html=True)
# elif authentication_status:
with st.sidebar:
    st.markdown("<p style='text-align: center; color: grey; '>" + img_to_html(banner, width=200) + "</p>",
                unsafe_allow_html=True)
    # if 'user' not in st.session_state:
    #     st.session_state.user = username
    st.markdown(f" <h1 style='text-align: center; color: #FF9B24; font-size:18px; "
                f"font-family:Avenir ;font-weight:normal;'>Hello, {Path.home()}!</h1> "
                , unsafe_allow_html=True)
    selected = option_menu(None, ['Home', 'Annotate', 'Analysis'],
                           icons=['house', 'pencil-square', 'file-earmark-arrow-up'],
                           menu_icon="cast", default_index=0, orientation="vertical",
                           styles={
                               "container": {"padding": "0!important", "background-color": "#fafafa"},
                               "icon": {"color": "black", "font-size": "18px"},
                               "nav-link": {"color": "black", "font-size": "16px", "text-align": "center",
                                            "margin": "0px",
                                            "--hover-color": "#eee"},

                               "nav-link-selected": {"font-size": "18px", "font-weight": "normal",
                                                     "color": "black", "background-color": "#FF9B24"},
                           }
                           )
    # _, midcol, _ = st.columns([0.5, 1, 0.5])
    # with midcol:
    #     authenticator.logout('Logout', 'main')

def navigation():
    if selected == 'Home':
        info.main()
    elif selected == 'Annotate':
        annotate.main()
    elif selected == 'Analysis':
        analysis.main()
    elif selected == None:
        info.load_view()
navigation()