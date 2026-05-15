"""
=============================================================================
  FlightSense — Previsão de Cancelamentos de Voos EUA 2024
  Grupo 1 — Rodrigo Ramos (a2023137922) | Bruno Almeida (a2023143583)
  Coimbra Business School | ISCAC — Ciência de Dados para a Gestão 2025/2026
=============================================================================
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FlightSense · Previsão de Cancelamentos",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#090E1A;--bg-card:#0F1729;--bg-el:#162035;
  --border:rgba(21,101,255,.18);--border-glow:rgba(0,212,255,.35);
  --blue:#1565FF;--cyan:#00D4FF;--amber:#F59E0B;
  --red:#EF4444;--green:#10B981;
  --tx:#E8EDF8;--muted:#6B7FA3;--dim:#3D506B;
  --mono:'JetBrains Mono',monospace;--sans:'DM Sans',sans-serif
}
.stApp{background:var(--bg)!important;font-family:var(--sans)!important}
.stApp::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(21,101,255,.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(21,101,255,.03) 1px,transparent 1px);
  background-size:40px 40px}
[data-testid="stSidebar"]{background:#080C18!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{font-family:var(--sans)!important;color:var(--tx)!important}
.block-container{padding:1.5rem 2.5rem 3rem!important;max-width:1400px}
h1,h2,h3{font-family:var(--mono)!important}
[data-testid="stMetric"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;
  border-radius:10px!important;padding:1rem 1.2rem!important;position:relative;overflow:hidden;transition:border-color .2s}
[data-testid="stMetric"]:hover{border-color:var(--border-glow)!important}
[data-testid="stMetric"]::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--blue),var(--cyan))}
[data-testid="stMetricLabel"] p{font-size:.72rem!important;font-weight:500!important;
  color:var(--muted)!important;text-transform:uppercase;letter-spacing:.06em;font-family:var(--mono)!important}
[data-testid="stMetricValue"]{font-size:1.7rem!important;font-weight:700!important;
  color:var(--tx)!important;font-family:var(--mono)!important;line-height:1.2!important}
[data-testid="stMetricDelta"]{font-size:.78rem!important;font-family:var(--mono)!important}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:1.5rem 0!important}
.js-plotly-plot .plotly .main-svg{background:transparent!important}
[data-testid="stPlotlyChart"]>div{background:transparent!important}
[data-testid="stForm"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;
  border-radius:14px!important;padding:1.6rem 1.8rem!important}
[data-testid="stFormSubmitButton"]>button{
  background:linear-gradient(135deg,var(--blue) 0%,#0D4FCC 100%)!important;
  border:1px solid rgba(21,101,255,.5)!important;color:white!important;
  font-family:var(--mono)!important;font-weight:600!important;font-size:.85rem!important;
  letter-spacing:.04em!important;border-radius:8px!important;padding:.65rem 1.4rem!important;
  transition:all .2s!important;box-shadow:0 0 20px rgba(21,101,255,.25)!important}
[data-testid="stFormSubmitButton"]>button:hover{
  box-shadow:0 0 32px rgba(0,212,255,.4)!important;border-color:var(--cyan)!important;
  transform:translateY(-1px)!important}
[data-testid="stTabs"] [role="tab"]{font-family:var(--mono)!important;font-size:.8rem!important;
  font-weight:500!important;color:var(--muted)!important;letter-spacing:.04em}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--cyan)!important;border-bottom-color:var(--cyan)!important}
[data-testid="stExpander"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important}
[data-testid="stAlert"][kind="error"]{background:rgba(239,68,68,.08)!important;border:1px solid rgba(239,68,68,.3)!important;border-radius:10px!important}
[data-testid="stAlert"][kind="success"]{background:rgba(16,185,129,.08)!important;border:1px solid rgba(16,185,129,.3)!important;border-radius:10px!important}
[data-testid="stMultiSelect"] span[data-baseweb="tag"]{
  background:rgba(21,101,255,.2)!important;border:1px solid rgba(21,101,255,.35)!important;
  color:#93C5FD!important;font-family:var(--mono)!important;font-size:.72rem!important;border-radius:4px!important}
[data-testid="stRadio"] [aria-checked="true"]+label,[data-testid="stRadio"] [aria-checked="true"]~div label{
  color:var(--cyan)!important;font-weight:600!important}
#MainMenu,footer,header{visibility:hidden!important}
[data-testid="stDecoration"]{display:none!important}
@media(min-width:769px){
  [data-testid="collapsedControl"]{display:none!important}
  [data-testid="stSidebarCollapseButton"]{display:none!important}
  section[data-testid="stSidebar"]{transform:translateX(0)!important;min-width:244px!important;width:244px!important}
  section[data-testid="stSidebar"][aria-expanded="false"]{margin-left:0!important;transform:translateX(0)!important;display:block!important}
}
@media(max-width:768px){
  [data-testid="stSidebarCollapseButton"]{display:flex!important}
  [data-testid="collapsedControl"]{
    display:flex!important;position:fixed!important;
    top:0.6rem!important;left:0.6rem!important;
    z-index:999999!important;background:rgba(21,101,255,0.95)!important;
    border-radius:8px!important;padding:0.4rem 0.6rem!important;
    box-shadow:0 2px 12px rgba(0,0,0,0.5)!important;
  }
  .block-container{padding:0.6rem 0.7rem 2rem!important}
  [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:0.5rem!important}
  [data-testid="column"]{min-width:calc(50% - 0.5rem)!important;flex:1 1 calc(50% - 0.5rem)!important}
  [data-testid="stForm"]{padding:0.8rem!important}
  [data-testid="stMetricValue"]{font-size:1.1rem!important}
  [data-testid="stMetricLabel"] p{font-size:.62rem!important}
  .ph h1{font-size:1.1rem!important}
  .ps{font-size:.72rem!important}
  .sl{font-size:.6rem!important}
  [data-testid="stTabs"] [role="tab"]{font-size:.7rem!important;padding:0.3rem 0.5rem!important}
  [data-testid="stPlotlyChart"]{overflow-x:auto!important}
}
@media(max-width:480px){
  [data-testid="column"]{min-width:100%!important;flex:1 1 100%!important}
}
.ph{border-left:3px solid var(--cyan);padding:.1rem 0 .1rem 1rem;margin-bottom:.3rem}
.ph h1{font-family:var(--mono)!important;font-size:1.55rem!important;font-weight:700!important;
  color:var(--tx)!important;letter-spacing:-.02em;margin:0!important}
.ps{font-size:.82rem;color:var(--muted);margin:.4rem 0 1.6rem 1.3rem;font-family:var(--sans)!important;letter-spacing:.01em}
.sl{font-family:var(--mono);font-size:.68rem;font-weight:700;color:var(--dim);letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:.8rem;padding-bottom:.5rem;border-bottom:1px solid var(--border)}
.chip{display:inline-block;font-family:var(--mono);font-size:.7rem;font-weight:600;
  padding:3px 10px;border-radius:99px;letter-spacing:.05em;margin-right:6px;margin-bottom:4px}
.cb{background:rgba(21,101,255,.15);color:#93C5FD;border:1px solid rgba(21,101,255,.3)}
.cc{background:rgba(0,212,255,.10);color:#67E8F9;border:1px solid rgba(0,212,255,.25)}
.cg{background:rgba(16,185,129,.12);color:#6EE7B7;border:1px solid rgba(16,185,129,.28)}
.cr{background:rgba(239,68,68,.12);color:#FCA5A5;border:1px solid rgba(239,68,68,.28)}
.ca{background:rgba(245,158,11,.12);color:#FCD34D;border:1px solid rgba(245,158,11,.28)}
.mb{background:var(--bg-el);border:1px solid var(--border);border-radius:8px;
  padding:.6rem 1rem;font-family:var(--mono);font-size:.75rem;color:var(--muted);
  display:flex;align-items:center;gap:12px;margin-bottom:1.2rem}
.mb strong{color:var(--cyan)}
.prow{display:flex;justify-content:space-between;padding:7px 0;
  border-bottom:1px solid rgba(21,101,255,.08);font-size:.83rem}
.prow:last-child{border-bottom:none}
.pk{color:var(--muted);font-family:var(--mono);font-size:.75rem}
.pv{color:var(--tx);font-weight:500}
.sf{font-family:var(--mono);font-size:.68rem;color:var(--dim);
  line-height:1.7;border-top:1px solid var(--border);padding-top:.8rem;margin-top:.4rem}
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
(function() {
  function keepSidebarOpen() {
    if (parent.window.innerWidth <= 768) return;
    var sidebar = parent.document.querySelector('section[data-testid="stSidebar"]');
    if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
      var btn = parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
      if (btn) btn.click();
    }
  }
  setInterval(keepSidebarOpen, 300);
})();
</script>
""", height=0)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
NOMES_MES = {1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
             7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"}
NOMES_DIA = {1:"Segunda-feira",2:"Terça-feira",3:"Quarta-feira",
             4:"Quinta-feira",5:"Sexta-feira",6:"Sábado",7:"Domingo"}
NOMES_AEROPORTO = {
  "ABI":"ABI — Abilene, TX","ABQ":"ABQ — Albuquerque, NM","ABR":"ABR — Aberdeen, SD",
  "ABY":"ABY — Albany, GA","ACT":"ACT — Waco, TX","ACV":"ACV — Arcata/Eureka, CA",
  "ACY":"ACY — Atlantic City, NJ","ADK":"ADK — Adak, AK","ADQ":"ADQ — Kodiak, AK",
  "AEX":"AEX — Alexandria, LA","AGS":"AGS — Augusta, GA","ALB":"ALB — Albany, NY",
  "ALW":"ALW — Walla Walla, WA","AMA":"AMA — Amarillo, TX","ANC":"ANC — Anchorage, AK",
  "APN":"APN — Alpena, MI","ASE":"ASE — Aspen, CO","ATL":"ATL — Atlanta, GA",
  "ATW":"ATW — Appleton, WI","AUS":"AUS — Austin, TX","AVL":"AVL — Asheville, NC",
  "AVP":"AVP — Wilkes-Barre/Scranton, PA","AZA":"AZA — Mesa (Phoenix), AZ",
  "AZO":"AZO — Kalamazoo, MI","BDL":"BDL — Hartford, CT","BET":"BET — Bethel, AK",
  "BFF":"BFF — Scottsbluff, NE","BFL":"BFL — Bakersfield, CA","BGM":"BGM — Binghamton, NY",
  "BGR":"BGR — Bangor, ME","BHM":"BHM — Birmingham, AL","BIH":"BIH — Bishop, CA",
  "BIL":"BIL — Billings, MT","BIS":"BIS — Bismarck, ND","BJI":"BJI — Bemidji, MN",
  "BLI":"BLI — Bellingham, WA","BLV":"BLV — Belleville, IL","BMI":"BMI — Bloomington, IL",
  "BNA":"BNA — Nashville, TN","BOI":"BOI — Boise, ID","BOS":"BOS — Boston, MA",
  "BPT":"BPT — Beaumont, TX","BQK":"BQK — Brunswick, GA","BQN":"BQN — Aguadilla, PR",
  "BRD":"BRD — Brainerd, MN","BRO":"BRO — Brownsville, TX","BRW":"BRW — Barrow, AK",
  "BTM":"BTM — Butte, MT","BTR":"BTR — Baton Rouge, LA","BTV":"BTV — Burlington, VT",
  "BUF":"BUF — Buffalo, NY","BUR":"BUR — Burbank, CA","BWI":"BWI — Baltimore, MD",
  "BZN":"BZN — Bozeman, MT","CAE":"CAE — Columbia, SC","CAK":"CAK — Akron/Canton, OH",
  "CDC":"CDC — Cedar City, UT","CDV":"CDV — Cordova, AK","CHA":"CHA — Chattanooga, TN",
  "CHO":"CHO — Charlottesville, VA","CHS":"CHS — Charleston, SC","CID":"CID — Cedar Rapids, IA",
  "CIU":"CIU — Sault Ste. Marie, MI","CKB":"CKB — Clarksburg, WV","CLE":"CLE — Cleveland, OH",
  "CLL":"CLL — College Station, TX","CLT":"CLT — Charlotte, NC","CMH":"CMH — Columbus, OH",
  "CMI":"CMI — Champaign, IL","CMX":"CMX — Hancock, MI","CNY":"CNY — Moab, UT",
  "COS":"COS — Colorado Springs, CO","COU":"COU — Columbia, MO","CPR":"CPR — Casper, WY",
  "CRP":"CRP — Corpus Christi, TX","CRW":"CRW — Charleston, WV","CSG":"CSG — Columbus, GA",
  "CVG":"CVG — Cincinnati, KY","CWA":"CWA — Wausau, WI","CYS":"CYS — Cheyenne, WY",
  "DAB":"DAB — Daytona Beach, FL","DAL":"DAL — Dallas Love Field, TX","DAY":"DAY — Dayton, OH",
  "DCA":"DCA — Washington Reagan, DC","DDC":"DDC — Dodge City, KS","DEC":"DEC — Decatur, IL",
  "DEN":"DEN — Denver, CO","DFW":"DFW — Dallas/Fort Worth, TX","DHN":"DHN — Dothan, AL",
  "DIK":"DIK — Dickinson, ND","DLH":"DLH — Duluth, MN","DRO":"DRO — Durango, CO",
  "DSM":"DSM — Des Moines, IA","DTW":"DTW — Detroit, MI","DVL":"DVL — Devils Lake, ND",
  "ECP":"ECP — Panama City Beach, FL","EGE":"EGE — Eagle/Vail, CO","EKO":"EKO — Elko, NV",
  "ELM":"ELM — Elmira, NY","ELP":"ELP — El Paso, TX","ESC":"ESC — Escanaba, MI",
  "EUG":"EUG — Eugene, OR","EVV":"EVV — Evansville, IN","EWR":"EWR — Newark, NJ",
  "EYW":"EYW — Key West, FL","FAI":"FAI — Fairbanks, AK","FAR":"FAR — Fargo, ND",
  "FAT":"FAT — Fresno, CA","FAY":"FAY — Fayetteville, NC","FCA":"FCA — Kalispell, MT",
  "FLG":"FLG — Flagstaff, AZ","FLL":"FLL — Fort Lauderdale, FL","FNT":"FNT — Flint, MI",
  "FOD":"FOD — Fort Dodge, IA","FSD":"FSD — Sioux Falls, SD","FSM":"FSM — Fort Smith, AR",
  "FWA":"FWA — Fort Wayne, IN","GCC":"GCC — Gillette, WY","GCK":"GCK — Garden City, KS",
  "GEG":"GEG — Spokane, WA","GFK":"GFK — Grand Forks, ND","GGG":"GGG — Longview, TX",
  "GJT":"GJT — Grand Junction, CO","GNV":"GNV — Gainesville, FL","GPT":"GPT — Gulfport, MS",
  "GRB":"GRB — Green Bay, WI","GRI":"GRI — Grand Island, NE","GRK":"GRK — Killeen, TX",
  "GRR":"GRR — Grand Rapids, MI","GSO":"GSO — Greensboro, NC","GSP":"GSP — Greenville, SC",
  "GTF":"GTF — Great Falls, MT","GTR":"GTR — Columbus/Starkville, MS","GUC":"GUC — Gunnison, CO",
  "GUM":"GUM — Guam, GU","HDN":"HDN — Hayden, CO","HGR":"HGR — Hagerstown, MD",
  "HHH":"HHH — Hilton Head, SC","HIB":"HIB — Hibbing, MN","HLN":"HLN — Helena, MT",
  "HNL":"HNL — Honolulu, HI","HOU":"HOU — Houston Hobby, TX","HPN":"HPN — White Plains, NY",
  "HRL":"HRL — Harlingen, TX","HSV":"HSV — Huntsville, AL","HTS":"HTS — Huntington, WV",
  "HYS":"HYS — Hays, KS","IAD":"IAD — Washington Dulles, DC","IAG":"IAG — Niagara Falls, NY",
  "IAH":"IAH — Houston Intercontinental, TX","ICT":"ICT — Wichita, KS","IDA":"IDA — Idaho Falls, ID",
  "ILM":"ILM — Wilmington, NC","IMT":"IMT — Iron Mountain, MI","IND":"IND — Indianapolis, IN",
  "INL":"INL — International Falls, MN","ISP":"ISP — Long Island/Islip, NY",
  "ITH":"ITH — Ithaca, NY","ITO":"ITO — Hilo, HI","JAC":"JAC — Jackson Hole, WY",
  "JAN":"JAN — Jackson, MS","JAX":"JAX — Jacksonville, FL","JFK":"JFK — New York JFK, NY",
  "JLN":"JLN — Joplin, MO","JMS":"JMS — Jamestown, ND","JNU":"JNU — Juneau, AK",
  "JST":"JST — Johnstown, PA","KOA":"KOA — Kailua-Kona, HI","KTN":"KTN — Ketchikan, AK",
  "LAN":"LAN — Lansing, MI","LAR":"LAR — Laramie, WY","LAS":"LAS — Las Vegas, NV",
  "LAW":"LAW — Lawton, OK","LAX":"LAX — Los Angeles, CA","LBB":"LBB — Lubbock, TX",
  "LBE":"LBE — Latrobe, PA","LBF":"LBF — North Platte, NE","LBL":"LBL — Liberal, KS",
  "LCH":"LCH — Lake Charles, LA","LCK":"LCK — Columbus, OH","LEX":"LEX — Lexington, KY",
  "LFT":"LFT — Lafayette, LA","LGA":"LGA — New York LaGuardia, NY","LGB":"LGB — Long Beach, CA",
  "LIH":"LIH — Lihue, HI","LIT":"LIT — Little Rock, AR","LNK":"LNK — Lincoln, NE",
  "LRD":"LRD — Laredo, TX","LSE":"LSE — La Crosse, WI","LWS":"LWS — Lewiston, ID",
  "MAF":"MAF — Midland, TX","MBS":"MBS — Saginaw, MI","MCI":"MCI — Kansas City, MO",
  "MCO":"MCO — Orlando, FL","MCW":"MCW — Mason City, IA","MDT":"MDT — Harrisburg, PA",
  "MDW":"MDW — Chicago Midway, IL","MEI":"MEI — Meridian, MS","MEM":"MEM — Memphis, TN",
  "MFE":"MFE — McAllen, TX","MFR":"MFR — Medford, OR","MGM":"MGM — Montgomery, AL",
  "MHK":"MHK — Manhattan, KS","MHT":"MHT — Manchester, NH","MIA":"MIA — Miami, FL",
  "MKE":"MKE — Milwaukee, WI","MLB":"MLB — Melbourne, FL","MLI":"MLI — Moline, IL",
  "MLU":"MLU — Monroe, LA","MOB":"MOB — Mobile, AL","MOT":"MOT — Minot, ND",
  "MQT":"MQT — Marquette, MI","MRY":"MRY — Monterey, CA","MSN":"MSN — Madison, WI",
  "MSO":"MSO — Missoula, MT","MSP":"MSP — Minneapolis, MN","MSY":"MSY — New Orleans, LA",
  "MTJ":"MTJ — Montrose, CO","MYR":"MYR — Myrtle Beach, SC","OAJ":"OAJ — Jacksonville, NC",
  "OAK":"OAK — Oakland, CA","OGG":"OGG — Kahului (Maui), HI","OKC":"OKC — Oklahoma City, OK",
  "OMA":"OMA — Omaha, NE","OME":"OME — Nome, AK","ONT":"ONT — Ontario, CA",
  "ORD":"ORD — Chicago O'Hare, IL","ORF":"ORF — Norfolk, VA","ORH":"ORH — Worcester, MA",
  "OTH":"OTH — North Bend, OR","OTZ":"OTZ — Kotzebue, AK","PAE":"PAE — Everett, WA",
  "PBG":"PBG — Plattsburgh, NY","PBI":"PBI — West Palm Beach, FL","PDX":"PDX — Portland, OR",
  "PGD":"PGD — Punta Gorda, FL","PHL":"PHL — Philadelphia, PA","PHX":"PHX — Phoenix, AZ",
  "PIA":"PIA — Peoria, IL","PIB":"PIB — Hattiesburg, MS","PIE":"PIE — St. Petersburg, FL",
  "PIH":"PIH — Pocatello, ID","PIT":"PIT — Pittsburgh, PA","PLN":"PLN — Pellston, MI",
  "PNS":"PNS — Pensacola, FL","PPG":"PPG — Pago Pago, AS","PRC":"PRC — Prescott, AZ",
  "PSC":"PSC — Pasco, WA","PSE":"PSE — Ponce, PR","PSG":"PSG — Petersburg, AK",
  "PSM":"PSM — Portsmouth, NH","PSP":"PSP — Palm Springs, CA","PVD":"PVD — Providence, RI",
  "PVU":"PVU — Provo, UT","PWM":"PWM — Portland, ME","RAP":"RAP — Rapid City, SD",
  "RDD":"RDD — Redding, CA","RDM":"RDM — Redmond, OR","RDU":"RDU — Raleigh-Durham, NC",
  "RFD":"RFD — Rockford, IL","RHI":"RHI — Rhinelander, WI","RIC":"RIC — Richmond, VA",
  "RIW":"RIW — Riverton, WY","RKS":"RKS — Rock Springs, WY","RNO":"RNO — Reno, NV",
  "ROA":"ROA — Roanoke, VA","ROC":"ROC — Rochester, NY","ROW":"ROW — Roswell, NM",
  "RST":"RST — Rochester, MN","RSW":"RSW — Fort Myers, FL","SAF":"SAF — Santa Fe, NM",
  "SAN":"SAN — San Diego, CA","SAT":"SAT — San Antonio, TX","SAV":"SAV — Savannah, GA",
  "SBA":"SBA — Santa Barbara, CA","SBN":"SBN — South Bend, IN","SBP":"SBP — San Luis Obispo, CA",
  "SCC":"SCC — Deadhorse, AK","SCE":"SCE — State College, PA","SCK":"SCK — Stockton, CA",
  "SDF":"SDF — Louisville, KY","SEA":"SEA — Seattle, WA","SFB":"SFB — Sanford, FL",
  "SFO":"SFO — San Francisco, CA","SGF":"SGF — Springfield, MO","SGU":"SGU — St. George, UT",
  "SHR":"SHR — Sheridan, WY","SHV":"SHV — Shreveport, LA","SIT":"SIT — Sitka, AK",
  "SJC":"SJC — San Jose, CA","SJT":"SJT — San Angelo, TX","SJU":"SJU — San Juan, PR",
  "SLC":"SLC — Salt Lake City, UT","SLN":"SLN — Salina, KS","SMF":"SMF — Sacramento, CA",
  "SMX":"SMX — Santa Maria, CA","SNA":"SNA — Orange County, CA","SPI":"SPI — Springfield, IL",
  "SPN":"SPN — Saipan, MP","SPS":"SPS — Wichita Falls, TX","SRQ":"SRQ — Sarasota, FL",
  "STC":"STC — St. Cloud, MN","STL":"STL — St. Louis, MO","STS":"STS — Santa Rosa, CA",
  "STT":"STT — St. Thomas, USVI","STX":"STX — St. Croix, USVI","SUN":"SUN — Sun Valley, ID",
  "SUX":"SUX — Sioux City, IA","SWF":"SWF — Newburgh, NY","SWO":"SWO — Stillwater, OK",
  "SYR":"SYR — Syracuse, NY","TLH":"TLH — Tallahassee, FL","TOL":"TOL — Toledo, OH",
  "TPA":"TPA — Tampa, FL","TRI":"TRI — Tri-Cities, TN","TTN":"TTN — Trenton, NJ",
  "TUL":"TUL — Tulsa, OK","TUS":"TUS — Tucson, AZ","TVC":"TVC — Traverse City, MI",
  "TWF":"TWF — Twin Falls, ID","TXK":"TXK — Texarkana, AR","TYR":"TYR — Tyler, TX",
  "TYS":"TYS — Knoxville, TN","USA":"USA — Concord, NC","VCT":"VCT — Victoria, TX",
  "VEL":"VEL — Vernal, UT","VLD":"VLD — Valdosta, GA","VPS":"VPS — Fort Walton Beach, FL",
  "WRG":"WRG — Wrangell, AK","XNA":"XNA — Fayetteville/Rogers, AR","XWA":"XWA — Williston, ND",
  "YAK":"YAK — Yakutat, AK","YUM":"YUM — Yuma, AZ",
}
C_RED="#EF4444"; C_GREEN="#10B981"; C_BLUE="#1565FF"; C_CYAN="#00D4FF"; C_AMBER="#F59E0B"
BG_CARD="#0F1729"
PL = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="DM Sans",color="#6B7FA3",size=11),
          title_font=dict(family="JetBrains Mono",color="#E8EDF8",size=13),
          legend=dict(bgcolor="rgba(15,23,41,.8)",bordercolor="rgba(21,101,255,.2)",borderwidth=1),
          margin=dict(t=50,b=40,l=20,r=20))

# Métricas reais — cancelamentos (HistGradient Boosting otimizado)
TAXA_BASE    = 1.53   # taxa de cancelamento pós-limpeza
TAXA_JAN     = 3.73   # taxa de cancelamento em Janeiro
AUC_ROC      = 0.854
F1_SCORE     = 0.151
RECALL       = 0.266
THRESHOLD    = 0.806  # TunedThresholdClassifierCV otimizado por F1

# Métricas reais — atrasos (XGBoost otimizado)
TAXA_BASE_DEL = 8.52
AUC_ROC_DEL   = 0.718
F1_SCORE_DEL  = 0.289
RECALL_DEL    = 0.396
THRESHOLD_DEL = 0.613

# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO
# ─────────────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent

def _load_model_safe(path):
    try:
        m = joblib.load(path)
        if isinstance(m, str) or (not hasattr(m,"predict_proba") and not hasattr(m,"predict")):
            st.error(f"Ficheiro de modelo inválido: {path}"); st.stop()
        return m
    except FileNotFoundError:
        st.error(f"{path} não encontrado."); st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar {path}: {e}"); st.stop()

@st.cache_resource(show_spinner="A inicializar modelos...")
def carregar_modelos():
    m_canc = _load_model_safe(_DIR / "modelo_cancelamentos_voos.pkl")
    m_del  = _load_model_safe(_DIR / "modelo_atrasos_voos.pkl")
    try:
        with open(_DIR / "feature_names.json") as f:
            features = [x for x in json.load(f) if x != "is_delayed"]
    except FileNotFoundError:
        st.error("feature_names.json não encontrado."); st.stop()
    return m_canc, m_del, features

modelo, modelo_del, feature_names = carregar_modelos()

# Dados do dashboard pré-agregados (evita carregar o CSV de 697MB em produção)
_DADOS = {
 'all': {'total':1041151,'canc':15905,'taxa':1.5276,'top_ap':'BRW','top_taxa':15.5172,
  'by_month':[{'month':1,'sum':13905,'count':540773,'taxa':2.5713},{'month':2,'sum':2000,'count':500378,'taxa':0.3997}],
  'by_week':[{'day_of_week':1,'sum':3138,'count':164293,'taxa':1.91},{'day_of_week':2,'sum':3298,'count':150995,'taxa':2.1842},{'day_of_week':3,'sum':1751,'count':154351,'taxa':1.1344},{'day_of_week':4,'sum':1611,'count':149688,'taxa':1.0762},{'day_of_week':5,'sum':2214,'count':148173,'taxa':1.4942},{'day_of_week':6,'sum':1758,'count':126933,'taxa':1.385},{'day_of_week':7,'sum':2135,'count':146718,'taxa':1.4552}],
  'by_dom':[{'day_of_month':1,'sum':151,'count':35655,'taxa':0.4235},{'day_of_month':2,'sum':82,'count':37432,'taxa':0.2191},{'day_of_month':3,'sum':165,'count':33961,'taxa':0.4859},{'day_of_month':4,'sum':333,'count':36286,'taxa':0.9177},{'day_of_month':5,'sum':130,'count':36442,'taxa':0.3567},{'day_of_month':6,'sum':343,'count':33207,'taxa':1.0329},{'day_of_month':7,'sum':479,'count':35342,'taxa':1.3553},{'day_of_month':8,'sum':529,'count':36827,'taxa':1.4364},{'day_of_month':9,'sum':967,'count':35026,'taxa':2.7608},{'day_of_month':10,'sum':451,'count':32238,'taxa':1.399},{'day_of_month':11,'sum':345,'count':36404,'taxa':0.9477},{'day_of_month':12,'sum':1178,'count':35923,'taxa':3.2792},{'day_of_month':13,'sum':1152,'count':30780,'taxa':3.7427},{'day_of_month':14,'sum':963,'count':33884,'taxa':2.842},{'day_of_month':15,'sum':1674,'count':36325,'taxa':4.6084},{'day_of_month':16,'sum':1329,'count':34810,'taxa':3.8179},{'day_of_month':17,'sum':716,'count':32974,'taxa':2.1714},{'day_of_month':18,'sum':475,'count':36904,'taxa':1.2871},{'day_of_month':19,'sum':692,'count':37446,'taxa':1.848},{'day_of_month':20,'sum':266,'count':33156,'taxa':0.8023},{'day_of_month':21,'sum':281,'count':36405,'taxa':0.7719},{'day_of_month':22,'sum':573,'count':37430,'taxa':1.5309},{'day_of_month':23,'sum':434,'count':35538,'taxa':1.2212},{'day_of_month':24,'sum':512,'count':33200,'taxa':1.5422},{'day_of_month':25,'sum':615,'count':37655,'taxa':1.6332},{'day_of_month':26,'sum':342,'count':37729,'taxa':0.9065},{'day_of_month':27,'sum':356,'count':33090,'taxa':1.0759},{'day_of_month':28,'sum':152,'count':36492,'taxa':0.4165},{'day_of_month':29,'sum':108,'count':19561,'taxa':0.5521},{'day_of_month':30,'sum':51,'count':16344,'taxa':0.312},{'day_of_month':31,'sum':61,'count':16685,'taxa':0.3656}],
  'top12':[{'origin':'BRW','sum':9,'count':58,'taxa':15.5172},{'origin':'DVL','sum':15,'count':108,'taxa':13.8889},{'origin':'JMS','sum':15,'count':108,'taxa':13.8889},{'origin':'PAE','sum':27,'count':223,'taxa':12.1076},{'origin':'ALW','sum':14,'count':116,'taxa':12.069},{'origin':'CMX','sum':12,'count':113,'taxa':10.6195},{'origin':'OME','sum':6,'count':58,'taxa':10.3448},{'origin':'OTZ','sum':6,'count':58,'taxa':10.3448},{'origin':'PLN','sum':10,'count':98,'taxa':10.2041},{'origin':'JST','sum':12,'count':118,'taxa':10.1695},{'origin':'MCW','sum':10,'count':100,'taxa':10.0},{'origin':'BIH','sum':9,'count':102,'taxa':8.8235}]},
 1: {'total':540773,'canc':13905,'taxa':2.5713,'top_ap':'PAE','top_taxa':23.2143,
  'by_month':[{'month':1,'sum':13905,'count':540773,'taxa':2.5713}],
  'by_week':[{'day_of_week':1,'sum':2834,'count':89572,'taxa':3.1639},{'day_of_week':2,'sum':2687,'count':83796,'taxa':3.2066},{'day_of_week':3,'sum':1653,'count':85159,'taxa':1.9411},{'day_of_week':4,'sum':1410,'count':73462,'taxa':1.9194},{'day_of_week':5,'sum':2038,'count':72819,'taxa':2.7987},{'day_of_week':6,'sum':1506,'count':63034,'taxa':2.3892},{'day_of_week':7,'sum':1777,'count':72931,'taxa':2.4365}],
  'by_dom':[{'day_of_month':1,'sum':17,'count':17264,'taxa':0.0985},{'day_of_month':2,'sum':25,'count':18976,'taxa':0.1317},{'day_of_month':3,'sum':19,'count':18520,'taxa':0.1026},{'day_of_month':4,'sum':49,'count':18048,'taxa':0.2715},{'day_of_month':5,'sum':17,'count':18108,'taxa':0.0939},{'day_of_month':6,'sum':327,'count':16890,'taxa':1.9361},{'day_of_month':7,'sum':458,'count':18650,'taxa':2.4558},{'day_of_month':8,'sum':498,'count':18314,'taxa':2.7192},{'day_of_month':9,'sum':949,'count':16454,'taxa':5.7676},{'day_of_month':10,'sum':437,'count':16675,'taxa':2.6207},{'day_of_month':11,'sum':331,'count':18376,'taxa':1.8013},{'day_of_month':12,'sum':1070,'count':17724,'taxa':6.037},{'day_of_month':13,'sum':686,'count':15156,'taxa':4.5263},{'day_of_month':14,'sum':948,'count':17122,'taxa':5.5367},{'day_of_month':15,'sum':1655,'count':17313,'taxa':9.5593},{'day_of_month':16,'sum':1239,'count':15658,'taxa':7.9129},{'day_of_month':17,'sum':640,'count':16552,'taxa':3.8666},{'day_of_month':18,'sum':433,'count':18496,'taxa':2.341},{'day_of_month':19,'sum':638,'count':18363,'taxa':3.4744},{'day_of_month':20,'sum':251,'count':15482,'taxa':1.6212},{'day_of_month':21,'sum':260,'count':18524,'taxa':1.4036},{'day_of_month':22,'sum':557,'count':18308,'taxa':3.0424},{'day_of_month':23,'sum':423,'count':16364,'taxa':2.5849},{'day_of_month':24,'sum':496,'count':16727,'taxa':2.9653},{'day_of_month':25,'sum':597,'count':18542,'taxa':3.2197},{'day_of_month':26,'sum':313,'count':18624,'taxa':1.6806},{'day_of_month':27,'sum':242,'count':15506,'taxa':1.5607},{'day_of_month':28,'sum':111,'count':18635,'taxa':0.5957},{'day_of_month':29,'sum':107,'count':18373,'taxa':0.5824},{'day_of_month':30,'sum':51,'count':16344,'taxa':0.312},{'day_of_month':31,'sum':61,'count':16685,'taxa':0.3656}],
  'top12':[{'origin':'PAE','sum':26,'count':112,'taxa':23.2143},{'origin':'PLN','sum':10,'count':50,'taxa':20.0},{'origin':'ALW','sum':11,'count':60,'taxa':18.3333},{'origin':'CMX','sum':10,'count':58,'taxa':17.2414},{'origin':'DVL','sum':9,'count':56,'taxa':16.0714},{'origin':'MCW','sum':8,'count':52,'taxa':15.3846},{'origin':'JST','sum':9,'count':62,'taxa':14.5161},{'origin':'JMS','sum':8,'count':56,'taxa':14.2857},{'origin':'DDC','sum':7,'count':51,'taxa':13.7255},{'origin':'DEC','sum':11,'count':81,'taxa':13.5802},{'origin':'FOD','sum':7,'count':52,'taxa':13.4615},{'origin':'JLN','sum':6,'count':51,'taxa':11.7647}]},
 2: {'total':500378,'canc':2000,'taxa':0.3997,'top_ap':'JMS','top_taxa':13.4615,
  'by_month':[{'month':2,'sum':2000,'count':500378,'taxa':0.3997}],
  'by_week':[{'day_of_week':1,'sum':304,'count':74721,'taxa':0.4068},{'day_of_week':2,'sum':611,'count':67199,'taxa':0.9092},{'day_of_week':3,'sum':98,'count':69192,'taxa':0.1416},{'day_of_week':4,'sum':201,'count':76226,'taxa':0.2637},{'day_of_week':5,'sum':176,'count':75354,'taxa':0.2336},{'day_of_week':6,'sum':252,'count':63899,'taxa':0.3944},{'day_of_week':7,'sum':358,'count':73787,'taxa':0.4852}],
  'by_dom':[{'day_of_month':1,'sum':134,'count':18391,'taxa':0.7286},{'day_of_month':2,'sum':57,'count':18456,'taxa':0.3088},{'day_of_month':3,'sum':146,'count':15441,'taxa':0.9455},{'day_of_month':4,'sum':284,'count':18238,'taxa':1.5572},{'day_of_month':5,'sum':113,'count':18334,'taxa':0.6163},{'day_of_month':6,'sum':16,'count':16317,'taxa':0.0981},{'day_of_month':7,'sum':21,'count':16692,'taxa':0.1258},{'day_of_month':8,'sum':31,'count':18513,'taxa':0.1674},{'day_of_month':9,'sum':18,'count':18572,'taxa':0.0969},{'day_of_month':10,'sum':14,'count':15563,'taxa':0.09},{'day_of_month':11,'sum':14,'count':18028,'taxa':0.0777},{'day_of_month':12,'sum':108,'count':18199,'taxa':0.5934},{'day_of_month':13,'sum':466,'count':15624,'taxa':2.9826},{'day_of_month':14,'sum':15,'count':16762,'taxa':0.0895},{'day_of_month':15,'sum':19,'count':19012,'taxa':0.0999},{'day_of_month':16,'sum':90,'count':19152,'taxa':0.4699},{'day_of_month':17,'sum':76,'count':16422,'taxa':0.4628},{'day_of_month':18,'sum':42,'count':18408,'taxa':0.2282},{'day_of_month':19,'sum':54,'count':19083,'taxa':0.283},{'day_of_month':20,'sum':15,'count':17674,'taxa':0.0849},{'day_of_month':21,'sum':21,'count':17881,'taxa':0.1174},{'day_of_month':22,'sum':16,'count':19122,'taxa':0.0837},{'day_of_month':23,'sum':11,'count':19174,'taxa':0.0574},{'day_of_month':24,'sum':16,'count':16473,'taxa':0.0971},{'day_of_month':25,'sum':18,'count':19113,'taxa':0.0942},{'day_of_month':26,'sum':29,'count':19105,'taxa':0.1518},{'day_of_month':27,'sum':114,'count':17584,'taxa':0.6483},{'day_of_month':28,'sum':41,'count':17857,'taxa':0.2296},{'day_of_month':29,'sum':1,'count':1188,'taxa':0.0842}],
  'top12':[{'origin':'JMS','sum':7,'count':52,'taxa':13.4615},{'origin':'DVL','sum':6,'count':52,'taxa':11.5385},{'origin':'DIK','sum':4,'count':50,'taxa':8.0},{'origin':'SBA','sum':30,'count':487,'taxa':6.1602},{'origin':'ALW','sum':3,'count':56,'taxa':5.3571},{'origin':'CDV','sum':3,'count':56,'taxa':5.3571},{'origin':'JST','sum':3,'count':56,'taxa':5.3571},{'origin':'PSG','sum':3,'count':56,'taxa':5.3571},{'origin':'XWA','sum':7,'count':135,'taxa':5.1852},{'origin':'ASE','sum':39,'count':834,'taxa':4.6763},{'origin':'CYS','sum':2,'count':52,'taxa':3.8462},{'origin':'CMX','sum':2,'count':55,'taxa':3.6364}]},
}

def _resolve_dados(meses_sel):
    if not meses_sel or set(meses_sel) == {1, 2}:
        return _DADOS['all']
    if meses_sel == [1] or meses_sel == (1,):
        return _DADOS[1]
    return _DADOS[2]

# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────
def construir_entrada(mes, dia_mes, dia_semana, aeroporto, distancia):
    """Constrói o vetor de features para o modelo de cancelamentos."""
    e = pd.DataFrame(0, index=[0], columns=feature_names)
    # Features derivadas automaticamente dos inputs
    vals = {
        "month":          mes,
        "day_of_month":   dia_mes,
        "day_of_week":    dia_semana,
        "distance":       distancia,
        "is_long_flight": int(distancia > 1500),
        "is_short_flight":int(distancia < 300),
        "is_weekend":     int(dia_semana in {6, 7}),
    }
    for col, val in vals.items():
        if col in e.columns:
            e[col] = val
    ap = f"origin_{aeroporto}"
    if ap in e.columns:
        e[ap] = 1
    return e.astype("float32")

def cor_risco(prob):
    return C_GREEN if prob < 0.4 else (C_AMBER if prob < 0.8 else C_RED)

def gauge_chart(prob, threshold):
    cor = cor_risco(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100, 1),
        title={"text":"", "font":{"color":"rgba(0,0,0,0)","size":1}},
        number={"suffix":"%","font":{"size":40,"family":"JetBrains Mono","color":"#E8EDF8"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#3D506B","tickwidth":1,
                       "tickfont":{"family":"JetBrains Mono","size":9,"color":"#6B7FA3"}},
               "bar":{"color":cor,"thickness":0.22},
               "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
               "steps":[{"range":[0,  40], "color":"rgba(16,185,129,.08)"},
                        {"range":[40, 80],  "color":"rgba(245,158,11,.08)"},
                        {"range":[80, 100], "color":"rgba(239,68,68,.08)"}],
               "threshold":{"line":{"color":"#00D4FF","width":2},"thickness":.7,
                            "value": threshold * 100}},
    ))
    skip = {"margin", "title_font"}
    fig.update_layout(height=230, margin=dict(t=20,b=10,l=30,r=30), title="",
                      **{k:v for k,v in PL.items() if k not in skip})
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:\'JetBrains Mono\';font-size:1.1rem;font-weight:700;'
                'color:#E8EDF8;margin-bottom:2px">✈ Flight<span style="color:#00D4FF">Sense</span></div>'
                '<div style="font-size:.73rem;color:#3D506B;font-family:\'DM Sans\';'
                'letter-spacing:.02em;margin-bottom:1rem">ML · Previsão de Cancelamentos EUA 2024</div>',
                unsafe_allow_html=True)
    st.divider()
    pagina = st.radio("nav", options=["  Dashboard","  Previsão","  Impacto","  Sobre"],
                      label_visibility="collapsed")
    st.divider()
    if "Dashboard" in pagina:
        st.markdown('<div class="sl">Filtros</div>', unsafe_allow_html=True)
        meses_sel  = st.multiselect("Meses", options=[1, 2], default=[1, 2],
                                    format_func=lambda m: NOMES_MES.get(m, str(m))[:3])
    else:
        meses_sel = None
    st.divider()
    st.markdown(f'<div class="sf">Grupo 1 · ISCAC 2025/2026<br>Rodrigo Ramos · Bruno Almeida<br>'
                f'<span style="color:#1565FF">HistGradient Boosting</span> · AUC {AUC_ROC}</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if "Dashboard" in pagina:
    st.markdown('<div class="ph"><h1>Dashboard de Cancelamentos</h1></div>'
                '<div class="ps">Análise exploratória · Dataset BTS 2024 · 1 041 151 voos comerciais EUA (Jan–Fev)</div>',
                unsafe_allow_html=True)

    d = _resolve_dados(meses_sel)
    total    = d["total"]; canc = d["canc"]; taxa = d["taxa"]
    top_ap   = d["top_ap"]; top_taxa = d["top_taxa"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Voos na amostra", f"{total:,}")
    k2.metric("Cancelados", f"{canc:,}", delta=f"{taxa:.2f}% da amostra")
    k3.metric("Taxa global", f"{taxa:.2f}%", delta=f"{taxa - TAXA_BASE:.2f}% vs. média")
    k4.metric("Aeroporto crítico", top_ap, delta=f"{top_taxa:.1f}% taxa")
    st.divider()

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<div class="sl">Taxa de cancelamento por mês</div>', unsafe_allow_html=True)
        tm = pd.DataFrame(d["by_month"])
        tm["mes"] = tm["month"].map(lambda m: NOMES_MES.get(int(m), str(m))[:3])
        fig_m = px.bar(tm, x="mes", y="taxa", color="taxa",
                       color_continuous_scale=[[0,"#162035"],[.4,"#1565FF"],[1,"#00D4FF"]],
                       text_auto=".2f")
        fig_m.update_traces(textposition="outside",
                            textfont=dict(family="JetBrains Mono", size=10, color="#E8EDF8"),
                            marker_line_width=0)
        fig_m.update_layout(**PL, coloraxis_showscale=False,
                            title="Taxa de cancelamento por mês",
                            xaxis=dict(title="", tickfont=dict(family="JetBrains Mono", size=10)),
                            yaxis=dict(title="Taxa (%)", gridcolor="rgba(21,101,255,.06)",
                                       tickfont=dict(family="JetBrains Mono", size=9)))
        st.plotly_chart(fig_m, use_container_width=True)

    with col_b:
        st.markdown('<div class="sl">Top 12 aeroportos — taxa de cancelamento</div>', unsafe_allow_html=True)
        t12 = pd.DataFrame(d["top12"])
        fig_a = px.bar(t12, x="taxa", y="origin", orientation="h", color="taxa",
                       color_continuous_scale=[[0,"#162035"],[.5,C_AMBER],[1,C_RED]],
                       text_auto=".2f")
        fig_a.update_traces(textposition="outside",
                            textfont=dict(family="JetBrains Mono", size=9, color="#E8EDF8"),
                            marker_line_width=0)
        fig_a.update_layout(**PL, coloraxis_showscale=False,
                            title="Top 12 aeroportos — taxa de cancelamento",
                            yaxis=dict(categoryorder="total ascending",
                                       tickfont=dict(family="JetBrains Mono", size=10, color="#E8EDF8")),
                            xaxis=dict(title="Taxa (%)", gridcolor="rgba(21,101,255,.06)",
                                       tickfont=dict(family="JetBrains Mono", size=9)))
        st.plotly_chart(fig_a, use_container_width=True)

    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        st.markdown('<div class="sl">Distribuição da variável alvo</div>', unsafe_allow_html=True)
        fig_p = go.Figure(go.Pie(
            values=[total - canc, canc], labels=["Operacional","Cancelado"], hole=.62,
            marker=dict(colors=[C_GREEN, C_RED], line=dict(color=BG_CARD, width=3)),
            textfont=dict(family="JetBrains Mono", size=11),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
        fig_p.add_annotation(
            text=f"<b>{taxa:.1f}%</b><br><span style='font-size:11px'>cancelados</span>",
            x=.5, y=.5, showarrow=False,
            font=dict(family="JetBrains Mono", size=18, color="#E8EDF8"))
        fig_p.update_layout(**{**PL, "title":"Distribuição da variável alvo",
                               "showlegend":True,
                               "legend":dict(orientation="h", y=-.1, x=.2)})
        st.plotly_chart(fig_p, use_container_width=True)

    with col_d:
        st.markdown('<div class="sl">Padrão semanal de cancelamentos</div>', unsafe_allow_html=True)
        td = pd.DataFrame(d["by_week"])
        td["dia"] = td["day_of_week"].map(lambda dw: NOMES_DIA.get(int(dw), str(dw))[:3])
        fig_d = go.Figure(go.Scatter(
            x=td["dia"], y=td["taxa"], mode="lines+markers",
            line=dict(color=C_CYAN, width=2.5, shape="spline"),
            marker=dict(size=8, color=C_CYAN, line=dict(color=BG_CARD, width=2)),
            fill="tozeroy", fillcolor="rgba(0,212,255,.06)",
            hovertemplate="<b>%{x}</b><br>Taxa: %{y:.2f}%<extra></extra>"))
        fig_d.update_layout(**PL, title="Padrão semanal de cancelamentos",
                            xaxis=dict(title="", tickfont=dict(family="JetBrains Mono", size=10)),
                            yaxis=dict(title="Taxa (%)", gridcolor="rgba(21,101,255,.06)",
                                       tickfont=dict(family="JetBrains Mono", size=9)))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown('<div class="sl">Taxa de cancelamento por dia do mês · feature mais importante (SHAP)</div>',
                unsafe_allow_html=True)
    tdom = pd.DataFrame(d["by_dom"])
    media_dom = tdom["taxa"].mean()
    fig_dom = go.Figure()
    fig_dom.add_hline(y=media_dom, line_dash="dash", line_color="#3D506B",
                      annotation_text=f"Média {media_dom:.2f}%",
                      annotation_font=dict(family="JetBrains Mono", size=9, color="#6B7FA3"))
    fig_dom.add_trace(go.Bar(
        x=tdom["day_of_month"], y=tdom["taxa"],
        marker=dict(color=tdom["taxa"],
                    colorscale=[[0,"#162035"],[0.4,"#1565FF"],[1,"#EF4444"]],
                    line=dict(width=0)),
        hovertemplate="Dia %{x}: %{y:.2f}%<extra></extra>"))
    fig_dom.update_layout(**PL, title="Taxa de cancelamento por dia do mês",
                          xaxis=dict(title="Dia do mês", dtick=1,
                                     tickfont=dict(family="JetBrains Mono", size=9)),
                          yaxis=dict(title="Taxa (%)", gridcolor="rgba(21,101,255,.06)",
                                     tickfont=dict(family="JetBrains Mono", size=9)))
    st.plotly_chart(fig_dom, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREVISÃO
# ─────────────────────────────────────────────────────────────────────────────
elif "Previsão" in pagina:
    st.markdown('<div class="ph"><h1>Previsão de Perturbações</h1></div>'
                '<div class="ps">Insere os dados do voo · os modelos respondem em tempo real</div>',
                unsafe_allow_html=True)

    aba_canc, aba_del = st.tabs(["  Cancelamento", "  Atraso"])

    # ── Formulário partilhado ──────────────────────────────────────────────
    def formulario(form_key):
        with st.form(form_key, clear_on_submit=False):
            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                st.markdown('<div class="sl">Período temporal</div>', unsafe_allow_html=True)
                mes        = st.selectbox("Mês do voo", options=[1, 2],
                                          format_func=lambda m: NOMES_MES[m], key=f"mes_{form_key}")
                dia_mes    = st.slider("Dia do mês", 1, 31, 15, key=f"dm_{form_key}")
                dia_semana = st.selectbox("Dia da semana", options=list(NOMES_DIA.keys()),
                                          format_func=lambda d: NOMES_DIA[d], key=f"ds_{form_key}")
            with c2:
                st.markdown('<div class="sl">Dados do voo</div>', unsafe_allow_html=True)
                aeroportos = sorted([c.replace("origin_","") for c in feature_names if c.startswith("origin_")])
                aeroporto  = st.selectbox("Aeroporto de origem", options=aeroportos,
                                          format_func=lambda ap: NOMES_AEROPORTO.get(ap, ap),
                                          key=f"ap_{form_key}")
                distancia  = st.number_input("Distância (milhas)", min_value=50, max_value=5000,
                                             value=800, step=50, key=f"dist_{form_key}")
            with c3:
                st.markdown('<div class="sl">Classificação automática</div>', unsafe_allow_html=True)
                curto  = distancia < 300
                longo  = distancia > 1500
                medio  = not curto and not longo
                fds    = dia_semana in {6, 7}
                def dot(on): return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{"#00D4FF" if on else "#3D506B"};margin-right:6px"></span>'
                st.markdown(
                    f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.15);'
                    f'border-radius:8px;padding:.8rem 1rem;font-family:\'JetBrains Mono\';'
                    f'font-size:.75rem;color:#6B7FA3">'
                    f'<div style="margin-bottom:6px">{dot(curto)}'
                    f'<span style="color:{"#00D4FF" if curto else "#3D506B"}">Curta distância</span>'
                    f'<span style="float:right">&lt; 300 mi</span></div>'
                    f'<div style="margin-bottom:6px">{dot(medio)}'
                    f'<span style="color:{"#93C5FD" if medio else "#3D506B"}">Médio curso</span>'
                    f'<span style="float:right">300–1500 mi</span></div>'
                    f'<div style="margin-bottom:8px">{dot(longo)}'
                    f'<span style="color:{"#FCD34D" if longo else "#3D506B"}">Longa distância</span>'
                    f'<span style="float:right">&gt; 1500 mi</span></div>'
                    f'<div style="padding-top:8px;border-top:1px solid rgba(21,101,255,.1)">'
                    f'{dot(fds)}<span style="color:{"#E8EDF8" if fds else "#3D506B"}">'
                    f'{"Fim de semana" if fds else "Dia útil"}</span></div></div>',
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("  Calcular probabilidade",
                                                  use_container_width=True, type="primary")
        return submitted, mes, dia_mes, dia_semana, aeroporto, distancia

    def mostrar_resultado(prob, pred, mes, dia_mes, dia_semana, aeroporto, distancia,
                          taxa_base, label_pos, label_neg, threshold, auc, f1, recall):
        st.divider()
        r1, r2 = st.columns([1, 1], gap="large")
        with r1:
            alto  = prob >= 0.8
            medio = 0.4 <= prob < 0.8
            lbl  = label_pos if alto else label_neg
            clr  = C_RED if alto else (C_AMBER if medio else C_GREEN)
            desc = ("O modelo identificou padrões históricos de risco elevado para este voo. "
                    "Probabilidade acima do limiar de alerta."
                    if alto else
                    ("Risco moderado identificado. Probabilidade acima da média global."
                     if medio else
                     "Sem padrões de risco elevado identificados. "
                     "Existe sempre risco residual não capturado pelo modelo."))
            brd = "rgba(239,68,68,.4)" if alto else ("rgba(245,158,11,.4)" if medio else "rgba(16,185,129,.4)")
            shd = "rgba(239,68,68,.08)" if alto else ("rgba(245,158,11,.08)" if medio else "rgba(16,185,129,.08)")
            st.markdown(
                f'<div style="background:#0F1729;border:1px solid {brd};box-shadow:0 0 24px {shd};'
                f'border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem">'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.7rem;color:{clr};'
                f'letter-spacing:.1em;margin-bottom:6px">{lbl}</div>'
                f'<div style="font-size:.85rem;color:#6B7FA3;line-height:1.55">{desc}</div></div>',
                unsafe_allow_html=True)

            st.markdown('<div class="sl">Parâmetros da previsão</div>', unsafe_allow_html=True)
            params = [
                ("Mês",           NOMES_MES[mes]),
                ("Dia do mês",    str(dia_mes)),
                ("Dia da semana", NOMES_DIA[dia_semana]),
                ("Fim de semana", "Sim" if dia_semana in {6,7} else "Não"),
                ("Aeroporto",     aeroporto),
                ("Distância",     f"{distancia:,} mi"),
                ("Voo curto",     "Sim" if distancia < 300 else "Não"),
                ("Voo longo",     "Sim" if distancia > 1500 else "Não"),
            ]
            rows = "".join(
                f'<div class="prow"><span class="pk">{k}</span><span class="pv">{v}</span></div>'
                for k, v in params)
            st.markdown(
                f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.15);'
                f'border-radius:10px;padding:.8rem 1.1rem">{rows}</div>',
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Média global", f"{taxa_base}%")
            rc2.metric("Este voo",     f"{prob*100:.1f}%",
                       delta=f"{prob*100 - taxa_base:+.1f}% vs. global")
            rc3.metric("Threshold",    str(threshold))

        with r2:
            st.plotly_chart(gauge_chart(prob, threshold), use_container_width=True)
            nivel = "ALTO" if prob >= 0.8 else ("MÉDIO" if prob >= 0.4 else "BAIXO")
            cn    = C_RED  if prob >= 0.8 else (C_AMBER if prob >= 0.4 else C_GREEN)
            pct   = prob * 100
            bars  = [
                ("rgba(16,185,129,.7)" if pct < 40  else "rgba(16,185,129,.15)"),
                ("rgba(245,158,11,.7)" if 40<=pct<80 else "rgba(245,158,11,.15)"),
                ("rgba(239,68,68,.7)"  if pct >= 80  else "rgba(239,68,68,.15)"),
            ]
            st.markdown(
                f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.15);'
                f'border-radius:10px;padding:1rem 1.2rem;margin-top:-.5rem">'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.68rem;color:#3D506B;'
                f'letter-spacing:.1em;margin-bottom:10px">NÍVEL DE RISCO</div>'
                f'<div style="display:flex;gap:6px;margin-bottom:10px">'
                + "".join(f'<div style="flex:1;height:6px;border-radius:99px;background:{b}"></div>'
                          for b in bars)
                + f'</div><span style="font-family:\'JetBrains Mono\';font-size:1.1rem;'
                f'font-weight:700;color:{cn}">{nivel}</span>'
                f'<span style="font-size:.8rem;color:#6B7FA3;margin-left:8px">'
                f'{pct:.1f}% probabilidade</span></div>',
                unsafe_allow_html=True)

        with st.expander("  Como interpretar este resultado?"):
            st.markdown(f"""
**O que significa {prob*100:.1f}%?**
Em voos com características semelhantes — {NOMES_MES[mes]}, {NOMES_DIA[dia_semana]},
partindo de {aeroporto} — o modelo estimou uma probabilidade de **{prob*100:.1f}%**.

**O threshold é {threshold}**, otimizado por validação cruzada (5 folds) para maximizar o F1-score.
Acima deste valor, o voo é classificado em risco. Esta escolha minimiza os falsos negativos —
casos em que um evento real passa despercebido.

**Desempenho do modelo:** AUC-ROC {auc} · F1 {f1} · Recall {recall}

**Limitações:** o modelo não tem acesso a dados meteorológicos em tempo real, estado da aeronave
ou decisões operacionais das companhias aéreas — os principais determinantes reais de cancelamentos e atrasos.
            """)

    # ── Aba Cancelamento ──────────────────────────────────────────────────
    with aba_canc:
        st.markdown(
            f'<div class="mb"><span>Modelo</span><strong>HistGradient Boosting</strong>'
            f'<span style="margin-left:auto">AUC-ROC <strong style="color:#00D4FF">{AUC_ROC}</strong></span>'
            f'<span>F1 <strong style="color:#F59E0B">{F1_SCORE}</strong></span>'
            f'<span>Recall <strong style="color:#E8EDF8">{RECALL}</strong></span>'
            f'<span>Threshold <strong style="color:#E8EDF8">{THRESHOLD}</strong></span></div>',
            unsafe_allow_html=True)
        sub_c, mes_c, dm_c, ds_c, ap_c, dist_c = formulario("canc")
        if sub_c:
            entrada = construir_entrada(mes_c, dm_c, ds_c, ap_c, dist_c)
            prob    = float(modelo.predict_proba(entrada)[0][1])
            pred    = int(modelo.predict(entrada)[0])
            mostrar_resultado(prob, pred, mes_c, dm_c, ds_c, ap_c, dist_c,
                              TAXA_BASE,
                              "RISCO DE CANCELAMENTO ELEVADO",
                              "VOO PROVAVELMENTE OPERACIONAL",
                              THRESHOLD, AUC_ROC, F1_SCORE, RECALL)

    # ── Aba Atraso ────────────────────────────────────────────────────────
    with aba_del:
        st.markdown(
            f'<div class="mb"><span>Modelo</span><strong>XGBoost</strong>'
            f'<span style="margin-left:auto">AUC-ROC <strong style="color:#00D4FF">{AUC_ROC_DEL}</strong></span>'
            f'<span>F1 <strong style="color:#F59E0B">{F1_SCORE_DEL}</strong></span>'
            f'<span>Recall <strong style="color:#E8EDF8">{RECALL_DEL}</strong></span>'
            f'<span>Threshold <strong style="color:#E8EDF8">{THRESHOLD_DEL}</strong></span></div>',
            unsafe_allow_html=True)
        st.info("Atraso definido como: atraso meteorológico ou por aeronave anterior >= 15 minutos, "
                "excluindo voos cancelados.", icon=None)
        sub_d, mes_d, dm_d, ds_d, ap_d, dist_d = formulario("del")
        if sub_d:
            entrada = construir_entrada(mes_d, dm_d, ds_d, ap_d, dist_d)
            prob    = float(modelo_del.predict_proba(entrada)[0][1])
            pred    = int(modelo_del.predict(entrada)[0])
            mostrar_resultado(prob, pred, mes_d, dm_d, ds_d, ap_d, dist_d,
                              TAXA_BASE_DEL,
                              "RISCO DE ATRASO ELEVADO",
                              "VOO PROVAVELMENTE PONTUAL",
                              THRESHOLD_DEL, AUC_ROC_DEL, F1_SCORE_DEL, RECALL_DEL)

# ─────────────────────────────────────────────────────────────────────────────
# SOBRE
# ─────────────────────────────────────────────────────────────────────────────
elif "Impacto" in pagina:
    st.markdown('<div class="ph"><h1>Impacto Operacional e Económico</h1></div>'
                '<div class="ps">Como o modelo FlightSense gera valor real para companhias aéreas e aeroportos</div>',
                unsafe_allow_html=True)

    # ── Contexto ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#0F1729;border:1px solid rgba(21,101,255,.15);border-radius:12px;
    padding:1.4rem 1.8rem;margin-bottom:1.5rem;line-height:1.75;color:#94A3B8;font-size:.9rem">
    Os modelos <b style="color:#E8EDF8">HistGradient Boosting</b> (cancelamentos) e
    <b style="color:#E8EDF8">XGBoost</b> (atrasos) foram treinados sobre mais de <b style="color:#00D4FF">1 milhão de voos</b>
    reais (EUA, 2024) e são capazes de sinalizar perturbações <b style="color:#E8EDF8">antes da partida</b>,
    sem aceder a dados meteorológicos em tempo real.<br><br>
    Cada cancelamento detetado antecipadamente permite à companhia aérea realocar tripulações,
    notificar passageiros e gerir gates com maior margem de tempo — reduzindo os custos
    operacionais associados a perturbações de última hora.
    </div>""", unsafe_allow_html=True)

    # ── Calculadora ───────────────────────────────────────────────────────────
    st.markdown('<div class="sl">Calculadora de Impacto Económico</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748B;font-size:.83rem;margin-bottom:1rem">Ajusta os parâmetros à realidade da tua companhia aérea para estimar o valor mensal e anual do modelo.</div>', unsafe_allow_html=True)

    col_inp, col_res = st.columns([1, 1], gap="large")

    with col_inp:
        st.markdown('<div class="sl">Parâmetros da companhia</div>', unsafe_allow_html=True)
        n_voos      = st.number_input("Voos por mês", min_value=100, max_value=100000, value=5000, step=100,
                                       help="Total de voos operados mensalmente")
        custo_canc  = st.number_input("Custo médio por cancelamento não antecipado (€)", min_value=1000, max_value=500000,
                                       value=45000, step=1000,
                                       help="Inclui compensações, realojamento, custos de tripulação e reputação")
        custo_atraso = st.number_input("Custo médio por atraso não antecipado (€)", min_value=500, max_value=100000,
                                        value=8000, step=500,
                                        help="Inclui combustível extra, taxas aeroportuárias e compensações")
        custo_fp    = st.number_input("Custo de um falso alerta (€)", min_value=0, max_value=5000,
                                       value=300, step=50,
                                       help="Custo de rever e descartar um alerta errado do modelo")

    with col_res:
        # Cálculos — cancelamentos
        canc_mes        = n_voos * (TAXA_BASE / 100)
        canc_detetados  = canc_mes * RECALL                          # recall=0.266
        fp_canc         = canc_detetados / 0.105 * (1 - 0.105)      # precision=0.105
        poupanca_canc   = canc_detetados * custo_canc - fp_canc * custo_fp

        # Cálculos — atrasos
        atrasos_mes      = n_voos * (TAXA_BASE_DEL / 100)
        atrasos_det      = atrasos_mes * RECALL_DEL                  # recall=0.396
        fp_del           = atrasos_det / 0.227 * (1 - 0.227)        # precision=0.227
        poupanca_atrasos = atrasos_det * custo_atraso - fp_del * custo_fp

        poupanca_total   = max(poupanca_canc, 0) + max(poupanca_atrasos, 0)
        poupanca_anual   = poupanca_total * 12

        st.markdown('<div class="sl">Estimativa mensal</div>', unsafe_allow_html=True)

        def kpi_box(label, value, sub, color="#00D4FF"):
            return (f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.15);'
                    f'border-radius:10px;padding:1rem 1.2rem;margin-bottom:.75rem">'
                    f'<div style="font-family:\'JetBrains Mono\';font-size:.68rem;color:#3D506B;'
                    f'letter-spacing:.08em;margin-bottom:.3rem">{label}</div>'
                    f'<div style="font-size:1.6rem;font-weight:700;color:{color};'
                    f'font-family:\'JetBrains Mono\'">{value}</div>'
                    f'<div style="font-size:.75rem;color:#64748B;margin-top:.2rem">{sub}</div></div>')

        st.markdown(
            kpi_box("CANCELAMENTOS DETETADOS / MÊS",
                    f"{canc_detetados:.0f}",
                    f"de {canc_mes:.0f} esperados · {RECALL*100:.0f}% recall") +
            kpi_box("POUPANÇA EST. — CANCELAMENTOS",
                    f"€ {max(poupanca_canc,0):,.0f}",
                    f"{canc_detetados:.0f} detetados × €{custo_canc:,} − {fp_canc:.0f} FP × €{custo_fp}",
                    color="#10B981") +
            kpi_box("ATRASOS DETETADOS / MÊS",
                    f"{atrasos_det:.0f}",
                    f"de {atrasos_mes:.0f} esperados · {RECALL_DEL*100:.0f}% recall") +
            kpi_box("POUPANÇA EST. — ATRASOS",
                    f"€ {max(poupanca_atrasos,0):,.0f}",
                    f"{atrasos_det:.0f} detetados × €{custo_atraso:,} − {fp_del:.0f} FP × €{custo_fp}",
                    color="#10B981"),
            unsafe_allow_html=True)

    # Total anual em destaque
    st.divider()
    ta1, ta2, ta3 = st.columns(3)
    ta1.metric("Poupança mensal estimada",   f"€ {poupanca_total:,.0f}")
    ta2.metric("Poupança anual estimada",    f"€ {poupanca_anual:,.0f}",
               delta=f"+{poupanca_anual/1e6:.2f}M €/ano" if poupanca_anual >= 1e6 else None)
    ta3.metric("Cancelamentos detetados/ano", f"{canc_detetados*12:.0f}",
               delta=f"{RECALL*100:.0f}% recall do modelo")
    st.divider()

    # ── Beneficiários ─────────────────────────────────────────────────────────
    st.markdown('<div class="sl">Valor por stakeholder</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3, gap="large")

    def beneficio_card(col, titulo, items, cor):
        with col:
            linhas = "".join(f'<li style="margin-bottom:.4rem">{i}</li>' for i in items)
            st.markdown(
                f'<div style="background:#0F1729;border:1px solid {cor};border-radius:12px;'
                f'padding:1.2rem 1.4rem;height:100%">'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.72rem;color:{cor};'
                f'letter-spacing:.08em;margin-bottom:.8rem">{titulo}</div>'
                f'<ul style="color:#94A3B8;font-size:.82rem;line-height:1.6;'
                f'padding-left:1.1rem">{linhas}</ul></div>',
                unsafe_allow_html=True)

    beneficio_card(b1, "COMPANHIAS AÉREAS", [
        "Realocação antecipada de tripulações e aeronaves",
        "Redução de custos de compensação de última hora",
        "Otimização do planeamento de slots",
        "Menor impacto em cascata na rede de voos",
    ], "#3B82F6")

    beneficio_card(b2, "AEROPORTOS", [
        "Gestão proativa de gates e recursos de solo",
        "Planeamento de pessoal baseado em risco",
        "Redução de congestionamento em dias críticos",
        "Informação antecipada sobre rotas de risco",
    ], "#06B6D4")

    beneficio_card(b3, "PASSAGEIROS", [
        "Notificação preventiva de perturbações",
        "Tempo para alternativas de viagem",
        "Menor tempo em espera sem informação",
        "Experiência de viagem mais previsível",
    ], "#10B981")

    st.divider()

    # ── Tabela comparativa ────────────────────────────────────────────────────
    st.markdown('<div class="sl">Sem modelo vs. com modelo FlightSense</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Situação": ["Cancelamentos antecipados", "Tempo de reação", "Custo médio por evento",
                     "Satisfação do passageiro", "Planeamento operacional"],
        "Sem Modelo": ["Reativo — detetado no momento",
                       "< 1 hora (impossível planear)",
                       f"€ {custo_canc:,} por cancelamento",
                       "Baixa — surpresa operacional",
                       "Baseado em histórico manual"],
        "Com FlightSense": [f"{RECALL*100:.0f}% detetados antecipadamente",
                            "Horas/dias de antecedência",
                            f"€ {custo_canc * (1 - RECALL * 0.7):,.0f} (redução estimada)",
                            "Alta — comunicação proativa",
                            "Orientado por dados e ML"],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)

    # ── Limitações honestas ───────────────────────────────────────────────────
    st.divider()
    with st.expander("  Limitações e pressupostos do modelo de impacto"):
        st.markdown("""
        - Os valores económicos são **estimativas** baseadas em benchmarks da indústria e nos parâmetros introduzidos — não são garantias de poupança.
        - O modelo tem **Recall de 26.6%** para cancelamentos: deteta 1 em cada 4, não todos.
        - Os **falsos positivos** (alertas incorretos) geram custos de revisão que foram incluídos no cálculo.
        - O impacto real depende da **velocidade de resposta operacional** após o alerta.
        - O modelo foi treinado apenas com dados de **2024 (EUA)** — padrões noutros contextos podem diferir.
        - Não inclui dados meteorológicos em tempo real — a principal fonte de melhoria futura.
        """)

elif "Sobre" in pagina:
    st.markdown('<div class="ph"><h1>Sobre o Projeto</h1></div>'
                '<div class="ps">FlightSense · Análise e Previsão de Cancelamentos EUA 2024</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom:1.5rem">'
                '<span class="chip cb">ISCAC · CBS</span>'
                '<span class="chip cc">Ciência de Dados para a Gestão</span>'
                '<span class="chip ca">2025/2026</span>'
                '<span class="chip cg">Projeto em CD</span></div>', unsafe_allow_html=True)

    cr, cb = st.columns(2, gap="large")
    def member_card(nome, num, role):
        return (f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.18);'
                f'border-radius:12px;padding:1.2rem 1.4rem">'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.7rem;color:#3D506B;'
                f'letter-spacing:.1em;margin-bottom:6px">INVESTIGADOR</div>'
                f'<div style="font-size:1rem;font-weight:600;color:#E8EDF8;margin-bottom:2px">{nome}</div>'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.75rem;color:#1565FF">{num}</div>'
                f'<div style="font-size:.8rem;color:#6B7FA3;margin-top:6px">{role}</div></div>')
    with cr:
        st.markdown(member_card("Rodrigo Ramos","a2023137922",
                                "Engenharia de Dados · Visualização · Notebooks Kaggle"),
                    unsafe_allow_html=True)
    with cb:
        st.markdown(member_card("Bruno Almeida","a2023143583",
                                "Documentação técnica · Modelação estatística · SHAP"),
                    unsafe_allow_html=True)

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registos (pós-limpeza)", "1 041 151")
    k2.metric("Features no modelo",     "340")
    k3.metric("Modelo final",           "HistGB Otimizado")
    k4.metric("AUC-ROC (teste)",        str(AUC_ROC))
    st.divider()

    t1, t2, t3, t4 = st.tabs(["📂  Dataset","⚙️  Metodologia","📈  Resultados","📑  Referências"])

    with t1:
        st.markdown("""
**Fonte:** [Kaggle · Flight Delay and Cancellation Data 2024](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024)
**Origem:** Bureau of Transportation Statistics (BTS) — U.S. DOT
**Período:** Janeiro e Fevereiro de 2024 · 1 041 151 registos após limpeza

| Variável | Tipo | Descrição |
|---|---|---|
| `month` | Numérico discreto | Mês do voo (1–2) |
| `day_of_month` | Numérico discreto | Dia do mês (1–31) |
| `day_of_week` | Numérico discreto | Dia da semana (1=Segunda … 7=Domingo) |
| `origin` | Categórico textual | Aeroporto de origem (código IATA) |
| `distance` | Numérico contínuo | Distância do voo em milhas |
| `is_long_flight` | Binário numérico | 1 se distância > 1 500 mi |
| `is_short_flight` | Binário numérico | 1 se distância < 300 mi |
| `is_weekend` | Binário numérico | 1 se sábado ou domingo |
| `cancelled` | Binário numérico | **Alvo** — 0 operacional / 1 cancelado |

**Desequilíbrio:** 98.47% operacionais · 1.53% cancelados (após limpeza)
        """)

    with t2:
        st.markdown("""
**M1 — Iniciação** · Objetivos SMART, perguntas de investigação, análise de viabilidade.

**M2 — Exploração e Preparação**
- *Missings* mantidos (associados a cancelamentos — informação estrutural, não erro)
- Remoção de 7 401 duplicados e 23 *outliers* físicos (velocidades supersónicas)
- Exclusão de variáveis com *data leakage*: `weather_delay`, `taxi_out`, `air_time`, etc.
- *Feature engineering*: `is_long_flight`, `is_short_flight`, `is_weekend`
- *One-Hot Encoding* em `origin` (~338 aeroportos · 340 features totais)
- Sem escalonamento — HistGradient Boosting e XGBoost são invariantes a transformações monotónicas

**M3 — Modelação**
- *Baseline*: Regressão Logística com `class_weight='balanced'`
- 3 candidatos comparados com *threshold* otimizado via curva Precisão-Recall
- `GridSearchCV` com `scoring='average_precision'` + `TunedThresholdClassifierCV`
- Divisão 80/20 estratificada · `StratifiedKFold` k=5 na validação cruzada
- Critério de seleção: **Average Precision** (PR-AUC) — mais honesta que ROC-AUC em classes desequilibradas
        """)

    with t3:
        res = pd.DataFrame({
            "Modelo": [
                "Regressão Logística (baseline)",
                "XGBoost",
                "HistGradient Boosting",
                "HistGB Otimizado ✓ (modelo final)",
            ],
            "F1-Score":      [0.087, 0.166, 0.163, 0.151],
            "Recall":        [0.233, 0.365, 0.214, 0.266],
            "Precision":     [0.053, 0.108, 0.132, 0.105],
            "AUC-ROC":       [0.761, 0.869, 0.867, 0.854],
            "Avg Precision": [0.042, 0.096, 0.096, 0.087],
            "Threshold":     [0.734, 0.785, 0.846, 0.806],
        })
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.markdown(f"""
- **HistGradient Boosting** selecionado com base na *Average Precision* (critério mais honesto para classes raras)
- *Threshold* {THRESHOLD} otimizado por validação cruzada (5 folds) · maximiza F1
- De 3 181 cancelamentos no teste, o modelo detetou **845** (Recall = {RECALL*100:.1f}%)
- Por cada cancelamento detetado, geram-se ~8.5 alertas preventivos (falsos positivos)
- Teto de desempenho limitado pela ausência de dados meteorológicos em tempo real
        """)

    with t4:
        st.markdown("""
**Dataset e fontes de dados:**
- **Nadeem, A. (2024).** *Flight Delay & Cancellation Data (1M+ 2024)*. Kaggle.
- **BTS (2024).** *Marketing Carrier On-Time Performance*. U.S. DOT. https://transtats.bts.gov

**Bibliotecas e metodologia:**
- **Pedregosa et al. (2011).** Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- **Chen & Guestrin (2016).** XGBoost: A scalable tree boosting system. *KDD 2016*.
- **Lundberg & Lee (2017).** A unified approach to interpreting model predictions. *NeurIPS 2017*.

---
**Docente:** Dora Melo — dmelo@iscac.pt
**Notebook Kaggle:** [Modelação — Previsão de Cancelamentos](https://www.kaggle.com/code/rodrigoramooos/modelacao-previsao-de-cancelamentos-em-voos)
        """)
