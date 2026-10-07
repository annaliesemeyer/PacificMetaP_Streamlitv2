import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statsmodels 
import matplotlib.cm as cm #colormap
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import seaborn as sns
import seaborn.objects as so
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.ticker as tkr
from adjustText import adjust_text
import matplotlib.patches as mpl_patches
from matplotlib import gridspec
from sklearn import linear_model
import plotly.express as px
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances 
from sklearn import preprocessing
import umap
import streamlit as st
from IPython.utils import io
from st_files_connection import FilesConnection

st.set_page_config(layout='wide')


plt.rcParams['font.size'] = 7
#plt.rcParams["font.weight"] = 'bold'
#plt.rcParams["axes.labelweight"] = 'bold'
mpl.rcParams['axes.linewidth'] = 1

# set tick width
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["axes.axisbelow"] = True

@st.cache_resource
def loadaws():
    conn = st.connection('s3', type=FilesConnection)
    taxa_filled_02 = conn.read("pacific-metap/GP15-17-OCE-02-3um_small_streamlit.csv", input_format="csv")
    taxa_filled_3 = conn.read("pacific-metap/GP15-17-OCE-3-51um_small_streamlit.csv", input_format="csv")
    return taxa_filled_02, taxa_filled_3
#taxa_filled_02 = pd.read_csv('GP15-17-OCE-02-3um.csv',low_memory=False)
#taxa_filled_3 = pd.read_csv('GP15-17-OCE-3-51um.csv',low_memory=False)

taxa_filled_02, taxa_filled_3 = loadaws()

sizefract = st.sidebar.radio('Size Fraction:',['0.2–3 µm', '3–51 µm'],index = 0)

@st.cache_data(ttl=3600)
def picksize(sizef):
    if sizef =='0.2–3 µm':
        taxa_domain_summed = pd.read_csv('Int_files_Streamlit/domain_02.csv')
        taxa_clade1_summed = pd.read_csv('Int_files_Streamlit/clade1_02.csv')
        taxa_clade2_summed = pd.read_csv('Int_files_Streamlit/clade2_02.csv')
        taxa_clade3_summed = pd.read_csv('Int_files_Streamlit/clade3_02.csv')
        taxa_clade4_summed = pd.read_csv('Int_files_Streamlit/clade4_02.csv')
        taxa_clade5_summed = pd.read_csv('Int_files_Streamlit/clade5_02.csv')
        taxa_genus_summed = pd.read_csv('Int_files_Streamlit/genus_02.csv')
        taxa_species_summed = pd.read_csv('Int_files_Streamlit/species_02.csv')
        #taxa_kegg_summed = pd.read_csv('Int_files_Streamlit/kegg_02.csv')
        #taxa_cog_summed = pd.read_csv('Int_files_Streamlit/cog_02.csv')
        taxa_protname_summed = pd.read_csv('Int_files_Streamlit/protname_02.csv')
        #taxa_path_summed = pd.read_csv('Int_files_Streamlit/path_02.csv')
        #taxa_EC_summed = pd.read_csv('Int_files_Streamlit/EC_02.csv')
        #taxa_PFAM_summed = pd.read_csv('Int_files_Streamlit/PFAM_02.csv')
        #taxa_module_summed = pd.read_csv('Int_files_Streamlit/module_02.csv')
    #    taxa_GO_summed = pd.read_csv('Int_files_Streamlit/GO_02.csv')
        #taxa_TC_summed = pd.read_csv('Int_files_Streamlit/TC_02.csv')
        taxa_filled = taxa_filled_02
        stn = pd.read_csv('GP15-17-OCE_stns.txt', sep='\t', encoding='latin-1')
        stn_names = pd.read_csv(r'Stn_Names_comma.csv')
        stn_ID = stn_names.Stn.astype(str)
        stn_ID = np.array(stn_ID)
        stn_keys = [1,3,4,5,6,8,9,10,12,14,15,16,18,18.3,19,21,23,25,27,29,31,33,35,37,39,1,3,6,8,10,12,14,16,18,20,22,25,27,29,32,33,35,37,38]
        stn_vals = ['GP15_02_1', 'GP15_02_3', 'GP15_02_4', 'GP15_02_5', 'GP15_02_6','GP15_02_8', 'GP15_02_9', 'GP15_02_10', 'GP15_02_12', 'GP15_02_14','GP15_02_15', 'GP15_02_16', 'GP15_02_18', 'GP15_02_18-3','GP15_02_19', 'GP15_02_21', 'GP15_02_23', 'GP15_02_25','GP15_02_27', 'GP15_02_29', 'GP15_02_31', 'GP15_02_33','GP15_02_35', 'GP15_02_37', 'GP15_02_39', 'GP17-OCE_02_1','GP17-OCE_02_3', 'GP17-OCE_02_6', 'GP17-OCE_02_8','GP17-OCE_02_10', 'GP17-OCE_02_12', 'GP17-OCE_02_14','GP17-OCE_02_16', 'GP17-OCE_02_18', 'GP17-OCE_02_20','GP17-OCE_02_22', 'GP17-OCE_02_25', 'GP17-OCE_02_27','GP17-OCE_02_29', 'GP17-OCE_02_32', 'GP17-OCE_02_33','GP17-OCE_02_35', 'GP17-OCE_02_37', 'GP17-OCE_02_38']
        stn_dict = dict(zip(stn_keys, stn_vals))
        
    
    if sizef =='3–51 µm':
        taxa_domain_summed = pd.read_csv('Int_files_Streamlit/domain_3.csv')
        taxa_clade1_summed = pd.read_csv('Int_files_Streamlit/clade1_3.csv')
        taxa_clade2_summed = pd.read_csv('Int_files_Streamlit/clade2_3.csv')
        taxa_clade3_summed = pd.read_csv('Int_files_Streamlit/clade3_3.csv')
        taxa_clade4_summed = pd.read_csv('Int_files_Streamlit/clade4_3.csv')
        taxa_clade5_summed = pd.read_csv('Int_files_Streamlit/clade5_3.csv')
        taxa_genus_summed = pd.read_csv('Int_files_Streamlit/genus_3.csv')
        taxa_species_summed = pd.read_csv('Int_files_Streamlit/species_3.csv')
        #taxa_kegg_summed = pd.read_csv('Int_files_Streamlit/kegg_3.csv')
        #taxa_cog_summed = pd.read_csv('Int_files_Streamlit/cog_3.csv')
        taxa_protname_summed = pd.read_csv('Int_files_Streamlit/protname_3.csv')
        #taxa_path_summed = pd.read_csv('Int_files_Streamlit/path_3.csv')
        #taxa_EC_summed = pd.read_csv('Int_files_Streamlit/EC_3.csv')
        #taxa_PFAM_summed = pd.read_csv('Int_files_Streamlit/PFAM_3.csv')
        #taxa_module_summed = pd.read_csv('Int_files_Streamlit/module_3.csv')
    #    taxa_GO_summed = pd.read_csv('Int_files_Streamlit/GO_3.csv')
        #taxa_TC_summed = pd.read_csv('Int_files_Streamlit/TC_3.csv')
        taxa_filled = taxa_filled_3  
        stn = pd.read_csv('GP15-17-OCE_stns3.txt', sep='\t', encoding='latin-1')
        stn3 = pd.read_csv('stnlabel_to_ID_3.txt', sep='\t', encoding='latin-1')
        stn_ID = stn3['Stn ID'].astype(str)
        stn_ID = np.array(stn_ID)
        stn_keys=[3,4,5,6,8,9,10,12,14,15,16,18,18.3,19,21,23,25,27,29,31,33,35,37,39,1,3,6,8,10,12,14,16,18,20,22,25,27,32,33,35,37,38]
        stn_vals = ['GP15_3_3','GP15_3_4','GP15_3_5','GP15_3_6','GP15_3_8','GP15_3_9','GP15_3_10','GP15_3_12','GP15_3_14','GP15_3_15','GP15_3_16','GP15_3_18','GP15_3_18-3','GP15_3_19','GP15_3_21','GP15_3_23','GP15_3_25','GP15_3_27','GP15_3_29','GP15_3_31','GP15_3_33','GP15_3_35','GP15_3_37','GP15_3_39','GP17-OCE_3_1','GP17-OCE_3_3','GP17-OCE_3_6','GP17-OCE_3_8','GP17-OCE_3_10','GP17-OCE_3_12','GP17-OCE_3_14','GP17-OCE_3_16','GP17-OCE_3_18','GP17-OCE_3_20','GP17-OCE_3_22','GP17-OCE_3_25','GP17-OCE_3_27','GP17-OCE_3_32','GP17-OCE_3_33','GP17-OCE_3_35','GP17-OCE_3_37','GP17-OCE_3_38']
        stn_dict3 = dict(zip(stn_keys, stn_vals))
#    return taxa_domain_summed, taxa_clade1_summed, taxa_clade2_summed, taxa_clade3_summed, taxa_clade4_summed, taxa_clade5_summed, taxa_genus_summed, taxa_species_summed, taxa_kegg_summed, taxa_cog_summed, taxa_protname_summed, taxa_path_summed, taxa_EC_summed, taxa_PFAM_summed, taxa_module_summed, taxa_TC_summed, taxa_filled, stn, stn_ID, stn_keys, stn_vals
    return taxa_domain_summed, taxa_clade1_summed, taxa_clade2_summed, taxa_clade3_summed, taxa_clade4_summed, taxa_clade5_summed, taxa_genus_summed, taxa_species_summed, taxa_protname_summed, taxa_filled, stn, stn_ID, stn_keys, stn_vals

taxa_domain_summed, taxa_clade1_summed, taxa_clade2_summed, taxa_clade3_summed, taxa_clade4_summed, taxa_clade5_summed, taxa_genus_summed, taxa_species_summed, taxa_protname_summed, taxa_filled, stn, stn_ID, stn_keys, stn_vals = picksize(sizefract)

st.markdown('# Pacific Metaproteomics Surface Transect')
st.markdown('#### Created by Annaliese Meyer')
st.markdown('Contact: acsmeyer@mit.edu')
st.markdown('These samples were collected as part of the GEOTRACES GP15 and GP17-OCE cruises. All data are presented as NSAF-normalized fractional abundances. 167 000 proteins were identified in the 0.2–3 µm size fraction, and 163 000 in the 3–51 µm fraction. You can select your size fraction of interest in the sidebar.')
st.markdown('These data are to be used for educational purposes only.')

@st.cache_resource
def loadimage():
    st.image('Methods.svg', width = 'stretch',output_format="PNG")

st.markdown('### Collection Methods')
loadimage()

st.markdown('## Community Composition')
################
optiontax = st.selectbox('Taxonomic Resolution:',['Domain', 'Supergroup', 'Phylum', 'Class','Order','Family','Genus','Species'],index = 0)

if optiontax =='Domain':
    data = taxa_domain_summed
if optiontax =='Supergroup':
    data = taxa_clade1_summed
if optiontax =='Phylum':
    data = taxa_clade2_summed
if optiontax =='Class':
    data = taxa_clade3_summed
if optiontax =='Order':
    data = taxa_clade4_summed
if optiontax =='Family':
    data = taxa_clade5_summed
if optiontax =='Genus':
    data = taxa_genus_summed
if optiontax =='Species':
    data = taxa_species_summed

df = pd.DataFrame(data)

df_sort = df.sort_values('sum')

# Create stacked bar plot
fig = px.bar(df, x='stn', y='sum', color='clade', barmode='stack',template="plotly_white", width=1500, height=800,labels = {'stn':'Station','sum':'Fractional Sum', 'clade':'Clade'}) #, color_discrete_map= clade4_dict
fig.update_traces(marker_line_width=0)
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, tickmode = "array", tickvals = stn_vals, ticktext = stn_keys)
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range = [0,1])
fig.update_layout(
    font_family="Arial", font_size = 14, autosize=True
)
#fig.update_layout(legend=dict(
#    orientation="h",
#    entrywidth=70,
#    yanchor="bottom",
#    y=1.02,
#    xanchor="right",
#    x=1
#))

st.plotly_chart(fig,use_container_width=True)

#############

taxselect = st.selectbox('Taxon:',list(data.clade.unique()))
col1, col2 = st.columns(2)
#con1 = col1.container(border=True)

with col1:
    data1 = data[data['clade'].str.contains(taxselect)]
    
    data2 = data1.groupby("stn").agg(
        lat = pd.NamedAgg(column="lat", aggfunc="min"),
        lon = pd.NamedAgg(column="lon", aggfunc="min"),
        summed = pd.NamedAgg(column="sum", aggfunc="sum"),
        stn = pd.NamedAgg(column="stn", aggfunc="min"),
        param_group = pd.NamedAgg(column="clade", aggfunc="sum")
    )
    
    
    fig = px.scatter_map(data2, lat = 'lat', lon = 'lon', color = 'summed',
                         hover_name="stn", size="summed",size_max = 30,color_continuous_scale="GnBu",opacity = 0.9)
    
    fig.update_layout(height=900, width = 600)
    fig.update_layout(coloraxis_colorbar_title_text = '% per station')
    fig.update_layout(
        autosize=False,
        hovermode='closest',
        map=dict(
            bearing=0,
            pitch=0,
            zoom=0,
            bounds=dict(
                west=-170,
                south=-70,
                east=-49,
                north=63),
            style="carto-darkmatter"
        ))
    
    fig.update_layout(font_size = 14, font_color = 'black'
        
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
    paper_bgcolor='indigo',
    margin=dict(pad=0, r=2, t=2, b=2, l=2)
)
    st.plotly_chart(fig,use_container_width=False)

with col2:
    st.markdown('The bubbles on this map show the relative amount of '+ str(taxselect) + ' in the '+str(sizefract)+' size fraction at each station along the transect. Hover over each point for the station name and the fractional contribution of that taxon to the proteins at that station.')
##############

#markertable = pd.read_csv('proteinmarkertable.csv')
#st.table(markertable)



markersinfo = '''
#### Iron
| Protein Name | Function |
| :- | :- |
fecA	| Fe(III) dicitrate transport protein
fecB | Fe complex transport system SBP
futA1/afuA	| Fe(III) transport system SBP
irpA/icmP	| Putative Fe regulated protein
ftn	| Ferritin - Fe storage protein
isiB/nifF/fldA |	Flavodoxin - electron transport protein
efeO	| Fe uptake system component
efeU	| High affinity Fe transporter
acr	| Fe complex outer membrane receptor
sir | Ferredoxin-sulfite reductase
ho1, pbsA1, hmuO |	Heme oxygenase
fiu	| Catecholate siderophore receptor
sfuA	| Fe(III) transport system SBP
bhuR/hemR/TbpA	| Hemoglobin/transferrin/lactoferrin receptor

#### Zinc
| Protein Name | Function |
| :- | :- |
ftsH	| Cell division metalloprotease
lraI	| Zn/Mn transport system SBP
COG0523 Cluster 1	| Putative Zn chaperone
znuA	| Zn transport system SBP
znuD	| Zn transport system ATP-binding protein
'''

markersinfo2 = '''
#### Vitamin B12
| Protein Name | Function |
| :- | :- |
nrdJ	| Ribonucleotide reductase, class II (B12-dependent)
nrdA/Z	| Ribonucleotide reductase, class Ia (B12-independent)
nrdB	| Ribonucleotide reductase, class Ia (B12-independent)
btuB	| B12 outer membrane transporter
metH	| Methionine synthase (B12-dependent)
metE	| Methionine synthase (B12-independent)
cUBN	| B12 outer membrane receptor
cobA	| B12 synthesis protein (cob(I)alamin adenosyltransferase)
cobS	| B12 synthesis protein (cobaltochelatase)

#### Nitrate
| Protein Name | Function |
| :- | :- |
urtA	| Urea transport system SBP
urtD	| Urea transport system ATP-binding protein
urtE	| Urea transport system ATP-binding protein
ureA	| Urease
ureC	| Urease
ntrP	| Nitrate/Nitrite transporter
ntcA	| Global nitrogen transcriptional regulator

#### Phosphate
| Protein Name | Function |
| :- | :- |
pstB	| Phosphate transport system ATP-binding protein
pstS	| Phosphate transport system SBP
phnD	| Phosphonate transport system SBP
sqdB	| Sulfolipid synthesis protein
PTC1	| Protein phosphatase
Acid Phosphatase	| Acid phosphatase
'''


####

st.markdown('## Functional Distribution')

protselect = st.selectbox('Select a protein or enter your own:',['fecA', 'cobW','irpA','metE','metH','ureC','urtA','ftsH','btuB'],accept_new_options=True, index = 0)
st.markdown('_Protein names are case sensitive._')

if st.checkbox('Protein Markers Reference'):
    st.markdown('### Nutrient Stress Protein Markers')
    
    col1,col2 = st.columns(2)
    with col1:
        st.markdown(markersinfo)
    with col2: 
        st.markdown(markersinfo2)
    
data = taxa_protname_summed#.groupby(taxa_protname_summed['protname'])

#data = data.get_group('fecA')

data1 = data[data['protname'].str.contains(protselect)]

data2 = data1.groupby("stn").agg(
    lat = pd.NamedAgg(column="lat", aggfunc="min"),
    lon = pd.NamedAgg(column="lon", aggfunc="min"),
    summed = pd.NamedAgg(column="sum", aggfunc="sum"),
    stn = pd.NamedAgg(column="stn", aggfunc="min"),
    param_group = pd.NamedAgg(column="protname", aggfunc="sum")
)


fig = px.scatter_map(data2, lat = 'lat', lon = 'lon', color = 'summed',
                     hover_name="stn", size="summed",size_max = 30,color_continuous_scale="GnBu",opacity = 0.9)

#fig = px.scatter_mapbox(data, lat = 'lat', lon = 'lon', color = "sum",
#                     hover_name="stn", size="sum",size_max = 30,color_continuous_scale="YlOrRd", opacity = 1)
#fig.update_geos(fitbounds="locations")

#zoom 4, height 4000, width 1400 for print, size max 40

fig.update_layout(height=900, width = 600)
fig.update_layout(coloraxis_colorbar_title_text = '% per station')
fig.update_layout(
    autosize=False,
    hovermode='closest',
    map=dict(
        bearing=0,
        pitch=0,
        zoom=0,
        bounds=dict(
        west=-170,
        south=-70,
        east=-49,
        north=63),
        style="carto-darkmatter",
        
    ))
fig.update_layout(
    font_family="Arial", font_size = 14, font_color = 'black')

fig.update_coloraxes(showscale=False)
fig.update_layout(
    paper_bgcolor='indigo',
    margin=dict(pad=0, r=2, t=2, b=2, l=2)
)


taxa_filled_small = taxa_filled[['Protein']] 

taxa_filled_small['protname'] = taxa_filled['Preferred_name']
taxa_filled_small['Description'] = taxa_filled['Description_y']
taxa_filled_small[['Domain', 'Supergroup', 'Phylum', 'Class','Order','Family','Genus','Species']] = taxa_filled[['domain','clade1','clade2','clade3','clade4','clade5','genus','species']]
taxa_filled_small = taxa_filled_small.fillna('Other')
taxa_filled_small[stn_ID] = taxa_filled[stn_ID]


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig,use_container_width=False)
with col2:
    st.markdown('The bubbles on this map show the relative amount of '+ str(protselect) + ' from any organism in the '+str(sizefract)+' size fraction at each station along the transect. Hover over each point for the station name and the fractional contribution of that protein to the total number of proteins detected at that station.')
    st.markdown('The predicted functional role of '+str(protselect)+' is '+str(taxa_filled_small['Description'][0]) + '.')


toplot = taxa_filled_small[taxa_filled_small['protname'].str.contains(protselect)] #ko:K16087, ko:K16091, ko:K09815,ko:K02077,ko:K11959
maxval = toplot[stn_ID].max().max()
xlabels = stn_keys
fig = plt.figure(figsize = (8, 3))
fig.patch.set_facecolor('black')
cmap = ListedColormap(['#01ff07','#fe01b1','#ff9408','#a9561e','#490648','#aa23ff','#13bbaf','#247afd','#ec2d01','#3f9b0b', 'gray','tan','palegreen'])

ax = plt.subplot()
ax.set_box_aspect(1/3)
ax.patch.set_facecolor('black')
ax.set_title(str(protselect), color = "white", size = 9, loc = 'left')
ax = pd.plotting.parallel_coordinates(toplot, cols = stn_ID, class_column = optiontax, colormap = cmap, axvlines = False, linewidth = 0.5)#color=toplot.colours)
ax.set_ylim([0,maxval])
ax.spines['top'].set_color("palegreen")
ax.spines['bottom'].set_color("palegreen")
ax.spines['left'].set_color("palegreen")
ax.spines['right'].set_color("palegreen")
ax.tick_params(axis='x', colors="#e8e8e8", labelsize = 5, size = 0)
ax.tick_params(axis='y', colors="palegreen", labelsize = 6, size = 0)
ax.grid(False)
ax.vlines(range(0,len(xlabels)),ymin = 0, ymax = maxval,color = "#e8e8e8", zorder = 0, alpha = 0.5, linewidth = 0.3)
with io.capture_output() as captured:
    ax.set_xticklabels(xlabels)
ax.set_xlabel('Station', color = 'white', size = 6)
ax.get_legend().remove()
fig.legend(bbox_to_anchor=(0.91, 0.8), loc='upper left',frameon = False, framealpha = 1, fontsize = 4.5, facecolor='k', labelcolor ='#e8e8e8', draggable = True, edgecolor = 'palegreen')
#fig.legend(loc='outside right center', frameon = True, framealpha = 1, fontsize = 4, facecolor='k', labelcolor ='#e8e8e8', draggable = True, edgecolor = 'palegreen')
#plt.text(0.1,0.1,str(protselect),size = 15, weight ='bold')
ax.ticklabel_format(scilimits=(0,0), axis = 'y')

#st.pyplot(fig, width = 'stretch')
fig.savefig('lines.svg', format="svg", bbox_inches = "tight")
#st.pdf('lines.pdf', height = 'stretch')

st.markdown('   ')
st.markdown('This plot shows the fractional abundance of '+str(protselect)+' in the ' + str(sizefract)+' size fraction across the transect. Each individual line corresponds to a distinct version of the detected protein from a specific species. The line colours correspond with your selected taxonomic rank, '+str(optiontax)+'.')

st.image('lines.svg', width = 'stretch',output_format="PNG")


##########################








##########################################
st.markdown('#### Protein Comparisons')
#compare proteins
col1, col2 = st.columns(2)

with col1:
    st.markdown('Target Protein 1')
    target = st.selectbox('Select protein 1 from dropdown or enter your own:',['fecA', 'cobW','irpA','metE','metH','ureC','urtA','ftsH','btuB'],accept_new_options=True, index = 3)
    st.markdown('_Protein names are case sensitive._')

with col2:
    st.markdown('Target Protein 2')
    target2 = st.selectbox('Select protein 2 from dropdown or enter your own:',['fecA', 'cobW','irpA','metE','metH','ureC','urtA','ftsH','btuB'],accept_new_options=True, index = 4)


params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

#target
data = taxa_protname_summed#.groupby(taxa_protname_summed['protname'])

data1 = data[data['protname'].str.contains(target)]

data2 = data1.groupby("stn").agg(
    lat = pd.NamedAgg(column="lat", aggfunc="min"),
    lon = pd.NamedAgg(column="lon", aggfunc="min"),
    summed = pd.NamedAgg(column="sum", aggfunc="sum"),
    station = pd.NamedAgg(column="stn", aggfunc="min"),
    param_group = pd.NamedAgg(column="protname", aggfunc="sum")
)

datagroup1 = data2.groupby('station')



#target2
dataa = taxa_protname_summed#.groupby(taxa_protname_summed['protname'])

datab = dataa[dataa['protname'].str.contains(target2)]

datac = datab.groupby("stn").agg(
    lat = pd.NamedAgg(column="lat", aggfunc="min"),
    lon = pd.NamedAgg(column="lon", aggfunc="min"),
    summed = pd.NamedAgg(column="sum", aggfunc="sum"),
    station = pd.NamedAgg(column="stn", aggfunc="min"),
    param_group = pd.NamedAgg(column="protname", aggfunc="sum")
)

datagroupa = datac.groupby('station')


fig = plt.figure()

fig.patch.set_facecolor('black')
ax = plt.subplot()
#ax.patch.set_alpha(0.0)
ax.patch.set_facecolor('black')

for n, k in enumerate(datagroup1.groups.keys()):
    datagroup2 = datagroup1.get_group(k)
    datagroupb = datagroupa.get_group(k)


    c = ax.scatter(datagroupb['summed'],datagroup2['summed'],c = datagroup2.lat, clim = (-67,60), s = 60, alpha = 1, cmap = 'GnBu', edgecolor = 'white')


cbar = plt.colorbar(c)
cbar.ax.tick_params(which='both', color='white', labelcolor='white')
cbar.ax.set_ylabel('Latitude (˚N)', color = 'white')
ax.grid()
ax.ticklabel_format(scilimits=(0,0), axis = 'both')
#ax.set_xlim(-0.5,100)
#ax.loglog()
ax.set_ylabel('$F_{protein 2}$', color = 'palegreen', size = 10)
ax.set_xlabel('$F_{protein 1}$',color = 'palegreen', size = 10)
ax.spines['top'].set_color("palegreen")
ax.spines['bottom'].set_color("palegreen")
ax.spines['left'].set_color("palegreen")
ax.spines['right'].set_color("palegreen")
ax.tick_params(axis='y', colors="palegreen", labelsize = 10)
ax.tick_params(axis='x', colors="palegreen", labelsize =10)

fig.savefig('comps.svg', format="svg", bbox_inches = "tight")
#st.pdf('comps.pdf', height = 'stretch')

col1, col2 = st.columns(2)

with col1:
    
    st.image('comps.svg', width = 'stretch',output_format="PNG")

with col2:
    st.markdown('This plot compares the fractional abundance of two proteins in the '+str(sizefract)+' size fraction. The latitude of each associated station is indicated by the marker shading.')
