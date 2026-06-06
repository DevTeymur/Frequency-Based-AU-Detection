ThisCVPRpaperistheOpenAccessversion,providedbytheComputerVisionFoundation.
Exceptforthiswatermark,itisidenticaltotheacceptedversion;
thefinalpublishedversionoftheproceedingsisavailableonIEEEXplore.
Do ImageNet-trained models learn shortcuts?
The impact of frequency shortcuts on generalization
ShunxinWang RaymondVeldhuis NicolaStrisciuglio
UniversityofTwente
{s.wang-2, r.n.j.veldhuis, n.strisciuglio}@utwente.nl
Abstract Table 1. Our method needs less computation times compared
to[39](usingResNet18onanNVIDIAA40GPU).Computational
time of [39] on ImageNet-1k is estimated from their ImageNet-
Frequencyshortcutsrefertospecificfrequencypatternsthat
10experiments,consideringthatitincreasesproportionallytothe
models heavily rely on for correct classification. Previ-
numberofclasses.
ous studies have shown that models trained on small im-
age datasets often exploit such shortcuts, potentially im-
Dataset Time(h)[39] Time(h)(Ours)
pairing their generalization performance. However, exist-
CIFAR-10 7.5 0.5
ingmethodsforidentifyingfrequencyshortcutsrequireex-
ImageNet-1k 8500(354days) 220(9.2days)
pensive computations and become impractical for analyz-
ing models trained on large datasets. In this work, we andgroundtruth.
propose the first approach to more efficiently analyze fre- However, there exist shortcuts in the Fourier domain,
quencyshortcutsatalargescale. WeshowthatbothCNN which are implicitly embedded in image data characteris-
and transformer models learn frequency shortcuts on Im- ticsandnoteasilydetectablebyvisualinspection[37,39].
ageNet. We also expose that frequency shortcut solutions Such shortcut solutions consist of small frequency subsets
can yield good performance on out-of-distribution (OOD) that are easy-to-learn and sufficient for models to achieve
test sets which largely retain texture information. How- high classification rate. Typically, they correspond to sim-
ever, these shortcuts, mostly aligned with texture patterns, plefeaturesliketextures,shapesorcolorsinthespatialdo-
hinder model generalization on rendition-based OOD test main. Wang, et al. [39] identified frequency shortcuts by
sets. These observations suggest that current OOD eval- retaining relevant frequencies to classification of a certain
uations often overlook the impact of frequency shortcuts class. The relevance of an individual frequency was mea-
on model generalization. Future benchmarks could thus suredbythelossvalueofthemodeltestedonimagesofthe
benefit from explicitly assessing and accounting for these classwiththatfrequencyremoved. Thus,theirapproachre-
shortcuts to build models that generalize across a broader quires computational time increasing proportionally to the
rangeofOODscenarios. Codesareavailableathttps: numberofclassesinadatasetandtheimageresolution.Yet,
//github.com/nis-research/hfss. frequencyshortcutsusuallyconsistofmultiplefrequencies.
Themethodin[39]mightoverlooksomeshortcutsasindi-
vidual frequency relevance to classification neglects joint
1.Introduction contribution of frequencies. Furthermore, the computa-
tional burden limits its applicability for analyzing shortcut
Superficialcorrelationsbetweendataandgroundtruth[5,9] learning behavior of models trained on large datasets (e.g.
can be learned by models to minimize training objectives ImageNet) with hundreds or thousands of classes. Several
with the least effort [34]. This learning behavior is called studies have shown that ImageNet-trained models are bi-
shortcut learning, which either harms the generalization asedtowardstextures[7,8]. However,thereisnoconcrete
performanceofmodelsorgivesanillusionofgoodgeneral- evidence of what causes this phenomenon, although [39]
izationabilitieswhenlearnedshortcutsarepresentinOOD largelyattributesittofrequencyshortcutlearning.
test sets [39]. Artifacts and visual cues that cause short- In this work, we propose the first method that enables
cut learning are recognizable by visual inspection of im- the uncovering of frequency shortcuts learned by models
ages,andtheirimpactonthetrainingofmodelscanbemit- trained on large-scale datasets (e.g. ImageNet-1k), for
igated through data selection or augmentation [5, 19, 27], analyzing learning behavior and explaining model gener-
i.e. counteracting the spurious correlations between data alization performance in different OOD scenarios. Our
25198

methodimprovescomputationalefficiencyasitappliespar- ates a false impression of good generalization when short-
allelclass-wiselosscomputationandhierarchicalsearchin cutsarepresentinOODtestsets [1,19,39].
| the Fourier | domain | (see      | Tab. | 1), compared |              | to [39]. | More-   |          |                 |     |            |     |           |       |     |
| ----------- | ------ | --------- | ---- | ------------ | ------------ | -------- | ------- | -------- | --------------- | --- | ---------- | --- | --------- | ----- | --- |
| over, our   | method | considers |      | the joint    | contribution |          | of fre- |          |                 |     |            |     |           |       |     |
|             |        |           |      |              |              |          |         | Shortcut | identification. |     | Uncovering |     | shortcuts | helps | un- |
quenciestoclassification,thusbeingmoreeffectiveinfind-
derstandingthegeneralizabilityofmodelsondifferentout-
| ingshortcuts. |     | Ourcontributionsare: |     |           |     |          |        |                 |     |               |       |            |        |            |        |
| ------------- | --- | -------------------- | --- | --------- | --- | -------- | ------ | --------------- | --- | ------------- | ----- | ---------- | ------ | ---------- | ------ |
|               |     |                      |     |           |     |          |        | of-distribution |     | (OOD)         | data, | explaining |        | why models | fail   |
| 1. We develop |     | a hierarchical       |     | frequency |     | shortcut | search |                 |     |               |       |            |        |            |        |
|               |     |                      |     |           |     |          |        | to generalize   |     | to or perform |       | well       | on OOD | data.      | It can |
(HFSS) method, which enables the analysis of shortcut provide shortcut-related prior knowledge to develop tech-
| learning            | in  | the Fourier | domain                       |     | on large | datasets | with |             |         |     |          |                |     |                |      |
| ------------------- | --- | ----------- | ---------------------------- | --- | -------- | -------- | ---- | ----------- | ------- | --- | -------- | -------------- | --- | -------------- | ---- |
|                     |     |             |                              |     |          |          |      | niques that | improve |     | model    | generalization |     | and robustness |      |
| varyingclasscounts. |     |             | Wereducecomputationaltimeand |     |          |          |      |             |         |     |          |                |     |                |      |
|                     |     |             |                              |     |          |          |      | performance | based   | on  | shortcut | mitigation     |     | [27, 38].      | How- |
improvetheeffectivenessatidentifyingshortcuts. ever, common approaches are limited to identifying vi-
| 2. We discover |     | that ImageNet-trained |     |     | models | (both | CNN |                    |     |           |     |      |          |           |     |
| -------------- | --- | --------------------- | --- | --- | ------ | ----- | --- | ------------------ | --- | --------- | --- | ---- | -------- | --------- | --- |
|                |     |                       |     |     |        |       |     | sually inspectable |     | shortcuts |     | such | as text, | watermark | and |
and transformer architectures) are subject to learn fre- color patches [19, 24, 27], using e.g. saliency maps [33].
| quency | shortcuts. |                | Different | from      | shortcuts      |     | formed by |                 |          |          |             |           |            |        |          |
| ------ | ---------- | -------------- | --------- | --------- | -------------- | --- | --------- | --------------- | -------- | -------- | ----------- | --------- | ---------- | ------ | -------- |
|        |            |                |           |           |                |     |           | Rather than     | directly |          | identifying | shortcuts |            | in the | data, in |
| visual | cues       | [9], frequency |           | shortcuts | (easy-to-learn |     | fea-      |                 |          |          |             |           |            |        |          |
|        |            |                |           |           |                |     |           | [2, 5] shortcut |          | features | present     | in        | individual | images | were     |
tures) lead to good performance on both in-distribution quantified by assessing how difficult they were for mod-
| (ID) | and OOD | tests | if they | do  | not block | models | from |               |     |         |     |          |     |           |        |
| ---- | ------- | ----- | ------- | --- | --------- | ------ | ---- | ------------- | --- | ------- | --- | -------- | --- | --------- | ------ |
|      |         |       |         |     |           |        |      | els to learn. | The | authors | in  | [28, 30, | 32] | uncovered | short- |
learningothersemantics(difficultfeatures).
|     |     |     |     |     |     |     |     | cut features | learned | within |     | representation |     | space. | The au- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------- | ------ | --- | -------------- | --- | ------ | ------- |
3. OurHFSSenablesamorecomprehensiveassessmentof
|       |                  |     |     |           |     |      |         | thors in | [23] | investigated | shortcut |     | learning | by analyzing |     |
| ----- | ---------------- | --- | --- | --------- | --- | ---- | ------- | -------- | ---- | ------------ | -------- | --- | -------- | ------------ | --- |
| model | generalizability |     | by  | analyzing | OOD | data | charac- |          |      |              |          |     |          |              |     |
therelationshipbetweensemanticconceptsusingaknowl-
teristics,specificallythepresenceofshortcuts. Inexist- edge graph, sharing a similar idea to [42]. Despite visual
| ing evaluation |     | frameworks, |     | frequency |     | shortcuts | do not |     |     |     |     |     |     |     |     |
| -------------- | --- | ----------- | --- | --------- | --- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
shortcutcues,therearenon-observableshortcutsinthefre-
| always     | impair | OOD            | generalization |                | performance. |     | This    |           |         |         |            |          |          |                   |     |
| ---------- | ------ | -------------- | -------------- | -------------- | ------------ | --- | ------- | --------- | ------- | ------- | ---------- | -------- | -------- | ----------------- | --- |
|            |        |                |                |                |              |     |         | quency    | domain, | which   | are        | embedded | in       | data characteris- |     |
| emphasizes |        | the importance |                | of considering |              | the | role of |           |         |         |            |          |          |                   |     |
|            |        |                |                |                |              |     |         | tics. The | work    | of [39] | identified |          | relevant | frequencies       | to  |
shortcutswhendesigningfutureOODevaluationbench-
|     |     |     |     |     |     |     |     | classification, |     | which | potentially | contain |     | shortcut informa- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ----- | ----------- | ------- | --- | ----------------- | --- |
marks. tion. As this technique measures the relevance of one fre-
quencyatatime,itrequirescomputationalcostsincreasing
2.Relatedworks
|          |           |     |        |           |          |     |           | proportionally |         | to the    | number | of classes |       | and image     | resolu- |
| -------- | --------- | --- | ------ | --------- | -------- | --- | --------- | -------------- | ------- | --------- | ------ | ---------- | ----- | ------------- | ------- |
|          |           |     |        |           |          |     |           | tion. Thus,    | limited | attention |        | has been   | given | to uncovering |         |
| Shortcut | learning. |     | Models | can learn | shortcut |     | solutions |                |         |           |        |            |       |               |         |
frequencyshortcutsinlargedatasets,withexistinganalyses
| based on | superficial | correlations |     | between |     | data and | ground |     |     |     |     |     |     |     |     |
| -------- | ----------- | ------------ | --- | ------- | --- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
primarilyfocusedondatasetscontainingasmallnumberof
| truth [9, | 11] to | optimize | training | objectives |     | with | the least |     |     |     |     |     |     |     |     |
| --------- | ------ | -------- | -------- | ---------- | --- | ---- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
effort. Thislearningbehaviorisduetosimplicity-bias[34], classes[2,37,39,42].
Existingmethodstoidentifyshortcutsarelimitedtoei-
| which is | caused | by inductive |     | biases | provided | by  | gradient |     |     |     |     |     |     |     |     |
| -------- | ------ | ------------ | --- | ------ | -------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
thermanualinspection[19]orinvolveatime-consumingal-
| descent | or components |     | like ReLUs |     | [35]. | Shortcuts | can be |     |     |     |     |     |     |     |     |
| ------- | ------------- | --- | ---------- | --- | ----- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
gorithm[37,39].Methodsquantifyingshortcutinformation
visualcuesinthedatalikesourcetagsandartificialmark-
oruncoveringshortcutfeaturesprimarilyaimatmitigating
| ers [19, | 27]. | For instance, |     | one fifth | of the | horse | images |     |     |     |     |     |     |     |     |
| -------- | ---- | ------------- | --- | --------- | ------ | ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
from Pascal VOC dataset were found to contain a source shortcut learning, without explaining how shortcuts in the
|     |     |     |     |     |     |     |     | dataimpactmodelgeneralizationperformance. |     |     |     |     |     | Wearethe |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- | --- | -------- | --- |
tag, onwhichmodelsrelyasadiscriminantfeaturetorec-
firsttoenablefrequencyshortcutanalysisofmodelstrained
| ognizehorses[19]. |     | Nexttothesevisualcues, |     |     |     | visionmod- |     |     |     |     |     |     |     |     |     |
| ----------------- | --- | ---------------------- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
onlarge-scaledatasetsandlinkthemtogeneralizationper-
| els are | also subject | to  | shortcuts | implicitly |     | existing | in the |     |     |     |     |     |     |     |     |
| ------- | ------------ | --- | --------- | ---------- | --- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
formanceindifferentOODsettings.
frequencydomain,whichmanifestassmallsetsoffrequen-
ciescontributingsignificantlytoimageclassificationperfor-
mance[37,39]. Thestudyin[39]explainswhymodelsex- OODevaluation. Thegeneralizationandrobustnessper-
hibit a textures-bias for classification [8] from a frequency formanceofvisionmodelsareusuallyevaluatedusingextra
| shortcut | perspective, |     | but does | not | provide | further | investi- |           |               |     |      |      |                |     |         |
| -------- | ------------ | --- | -------- | --- | ------- | ------- | -------- | --------- | ------------- | --- | ---- | ---- | -------------- | --- | ------- |
|          |              |     |          |     |         |         |          | data that | is considered |     | OOD, | e.g. | data collected | at  | differ- |
gationonhowfrequencyshortcutsaffectthegeneralization ent time points [31], with different styles or renditions [8,
and robustness performance of ImageNet-trained models, 14, 36], with synthetic corruption effects [13, 17, 25, 40],
duetocomputationalburden. and with adversarial noise [4, 12, 20, 43]. However, such
Relyingonsimpleshortcutsolutionscouldharmthegen- benchmarksdonotconsidertheimpactofshortcutslearned
eralizationandrobustnessperformanceofmodels,asshort- by models and present in OOD test sets. This might ig-
cut learning appearing in early training might block mod- norecriticalfactorsrelatedtothegeneralizationandrobust-
els from learning other semantics related to the tasks at nesscapabilitiesofmodels.Theworkin[7]exploresmodel
hand[3,16,39,41]. However,suchreliancecouldalsocre- generalizationfromtheperspectivesofbiases,e.g. texture,
25199

Start
Select randomly a
No
|     |     | Stage 1? |     |     | subset from  |     |     |     |     | Top-N subsets of each class  |     |     |     |
| --- | --- | -------- | --- | --- | ------------ | --- | --- | --- | --- | ---------------------------- | --- | --- | --- |
previous stage
Yes
|     |     |     |     |     |     |     |     | Training images for evaluation |     |     |     | No  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | --- | --- |
Frequency patches generation
Decrease patch size
Final stage?
|     | Sample p% patches (from a subset from  |     |     |     |     |     |     | Process images by retaining   |     |     |     |     |     |
| --- | -------------------------------------- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | --- |
previous stage if available) for B times  each frequency subset  Yes
|     |         |     |      |     |     |     |     | DFM-filtered images  |     |     | DFM of each class |     |     |
| --- | ------- | --- | ---- | --- | --- | --- | --- | -------------------- | --- | --- | ----------------- | --- | --- |
|     | Stage 1 |     | …... |     |     |     |     |                      |     |     |                   |     |     |
…...
Stage 2
|     | ...     |     |      |     |     |     |     | Feed to a trained model  |     |     |     |     |     |
| --- | ------- | --- | ---- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- |
|     | Stage S |     | …... |     |     |     |     |                          |     |     |     |     |     |
each stage has a different
frequency resolution
|     |     | B frequency subsets  |     |     |     |     | Select Top-N subsets by loss values |     |     |     |     | End |     |
| --- | --- | -------------------- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | --- |

1. Sampling B frequency subsets 2. Evaluating shortcut information 3. Forwarding subsets for next stage
Figure1. SchemeofHFSS.Startingfromstage2, wesamplefrequencypatchesfromarandomfrequencysubsetsearchedinprevious
stage.Thisconfinesthesizeofsearchspace.Thewhitepatchesinthebinarymasksindicatesampledfrequencypatches.
| shape and | spectral | biases. | In this | work, | we  | investigate | the |     |     |     |     |     |     |
| --------- | -------- | ------- | ------- | ----- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
impactoffrequencyshortcutsonmodelgeneralizationabil-
+
ities,establishingconnectionsamongthesedifferentbiases.
Our work provides insights into when frequency shortcuts Generated by
Evenly generated
yieldgoodperformanceorareharmfultomodelgeneraliza- shifted window
tionandrobustnessperformance.
p% patches
3.Method
We propose a method to identify (non-visible) frequency Figure2. Thefrequencyspectrumisseparatedintopatches,with
shortcuts linked to intrinsic data characteristics rather than p%sampledforshortcutevaluation.
visual cues as in [27]. We reduce significantly computa- frequency combinations that potentially contain frequency
| tional demands |     | compared | to [39]. | Our | method | is  | the first |            |               |     |          |              |           |
| -------------- | --- | -------- | -------- | --- | ------ | --- | --------- | ---------- | ------------- | --- | -------- | ------------ | --------- |
|                |     |          |          |     |        |     |           | shortcuts. | We use random |     | sampling | as it allows | to easily |
solutionforshortcutidentificationthatenablestheanalysis consider joint contribution of frequencies to classification,
| ofmodels | trainedondatasets |        | witha     | largenumberof |     |           | sam- |                |                   |     |              |          |               |
| -------- | ----------------- | ------ | --------- | ------------- | --- | --------- | ---- | -------------- | ----------------- | --- | ------------ | -------- | ------------- |
|          |                   |        |           |               |     |           |      | improving      | the effectiveness |     | of frequency | shortcut | identifi-     |
| ples and | classes.          | In the | following | sections,     |     | we detail | our  |                |                   |     |              |          |               |
|          |                   |        |           |               |     |           |      | cation. Random | sampling          |     | has been     | shown    | to contribute |
measurement of shortcut learning in models and examine tostablesearchresultsindataaugmentationandreinforce-
theimpactofshortcutsongeneralization.
mentlearning[21,26,29],whilerequiringlowercomputa-
|     |     |     |     |     |     |     |     | tionsw.r.t. | optimization-basedmethods. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | -------------------------- | --- | --- | --- | --- |
3.1.Identifyingfrequencyshortcuts
|            |     |               |              |     |           |     |        | There              | are numerous | combinations |           | of frequencies | by       |
| ---------- | --- | ------------- | ------------ | --- | --------- | --- | ------ | ------------------ | ------------ | ------------ | --------- | -------------- | -------- |
| We propose | a   | method called | hierarchical |     | frequency |     | short- |                    |              |              |           |                |          |
|            |     |               |              |     |           |     |        | sampling frequency | components   |              | directly. | Instead        | of doing |
cut search (HFSS), which exploits hierarchical search of an exhaustive search, we apply hierarchical search in the
| frequency | subsets | in the | image | Fourier | spectrum. |     | This re- |                  |                                    |     |     |     |     |
| --------- | ------- | ------ | ----- | ------- | --------- | --- | -------- | ---------------- | ---------------------------------- | --- | --- | --- | --- |
|           |         |        |       |         |           |     |          | Fourierspectrum: | wedividetheFourierspectrumintofre- |     |     |     |     |
duces computational time compared to exhaustive search quencypatchesandsamplethesefrequencypatchestogen-
| strategies, | e.g. | [38, 39]. | The search | is  | separated | into | sev- |                  |              |     |         |              |          |
| ----------- | ---- | --------- | ---------- | --- | --------- | ---- | ---- | ---------------- | ------------ | --- | ------- | ------------ | -------- |
|             |      |           |            |     |           |      |      | erate candidates | of frequency |     | subsets | that contain | shortcut |
eralstages,eachstagegraduallynarrowingdownthespec-
|     |     |     |     |     |     |     |     | information,asillustratedinFig.2. |     |     |     | Weexploitoverlapping |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | -------------------- | --- |
trumsearchspaceinacoarse-to-finemanneranddetecting shiftedwindowsinverticalandhorizontaldirections[22]to
frequencysubsetsthatmodelsstronglyrelyonforclassifi-
preparefrequencypatchesandthusavoidbordereffects,in
cation. TheschemeofHFSSisshowninFig.1. addition to evenly separating the Fourier spectrum. HFSS
consistsofmultiplesearchstages,thenumberofwhichde-
Hierarchicalsearchforfrequencyshortcuts. Inorderto pendsonimageresolutionanddesiredfrequencyresolution
discoverthefrequencysubsetsthatmodelsrelyheavilyon of the identified frequency subsets. Lower image or fre-
forcorrectclassificationofeachclass,wesampledifferent quencyresolutionrequiresfewersearchstages.
25200

| In summary, |     | the overall | process | of  | HFSS | is as follows. |     | 1.0 |     |     |     |     |     |     |
| ----------- | --- | ----------- | ------- | --- | ---- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
AvgTPRsct
Each stage of HFSS consists of three steps, namely (1) 0.8 AvgTPRnon
|     |     |     |     |     |     |     |     | 0.8 |     |     |     |         | sct      |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | --- |
|     |     |     |     |     |     |     |     |     |     |     |     | AvgTPRD | sc F t M |     |
samplingBfrequencysubsetswhichcontainp%frequency RPTgvA 0.6 0.6 AvgTPRD F M
no n sct
| patches, | (2) evaluating |     | shortcut | information, |     | and (3) for- |     |     |     |     |     |     |     |     |
| -------- | -------------- | --- | -------- | ------------ | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
0.4 0.4
| warding | subsets | for next | stage. | At the | initial | search stage, |     |     |     |     |     |     |     |     |
| ------- | ------- | -------- | ------ | ------ | ------- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
0.2
0.2
| we sample | frequency |     | patches with | a large | size | and evalu- |     |     |     |     |     |     |     |     |
| --------- | --------- | --- | ------------ | ------- | ---- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
0.0 0.0
ate their shortcut information. We evaluate the frequency 0.0 0.2 0.2 0.4 0.4 0.6 0.6 0.8 0.8 1.0
t
subsets according to their contributions to classification: Figure 3. ResNet18 tested on original test images and DFM-
we process a subset of training images by retaining a fre- filteredimagesofImageNet-1k.BluelineshowstheaverageTPR
quencysubsetandmeasuretheclass-wiselossofamodel. ofclassessubjecttoshortcutsontheoriginaltestimages,andthe
orangelineshowstheresultsofnon-shortcutclasses,atdifferent
Frequencysubsetsthatcontributetoalowerlossvalueare
|                |            |     |          |           |           |            | thresholdt.     | Thegreenandredlinescorrespondtoresultstested |         |       |             |      |           |     |
| -------------- | ---------- | --- | -------- | --------- | --------- | ---------- | --------------- | -------------------------------------------- | ------- | ----- | ----------- | ---- | --------- | --- |
| stronger       | candidates | to  | indicate | frequency | shortcuts | as they    |                 |                                              |         |       |             |      |           |     |
|                |            |     |          |           |           |            | on DFM-filtered |                                              | images. | Lower | t indicates | weak | shortcuts | and |
| are sufficient | for        | the | model to | achieve   | a high    | prediction |                 |                                              |         |       |             |      |           |     |
highertsignifiesstrongerones.Thesizeofeachpointreflectsthe
| score. We | rank | frequency | subsets | by  | the loss | values and |     |     |     |     |     |     |     |     |
| --------- | ---- | --------- | ------- | --- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
numberofclasses.Thelargerthesize,themoreclassesincluded.
| usethetop-N    | masksofeachclassinthenextsearchstage. |            |        |        |           |             |         |              |     |          |          |           |     |        |
| -------------- | ------------------------------------- | ---------- | ------ | ------ | --------- | ----------- | ------- | ------------ | --- | -------- | -------- | --------- | --- | ------ |
| Starting       | from                                  | the second | search | stage, | we        | sample fre- |         |              |     |          |          |           |     | t      |
|                |                                       |            |        |        |           |             | for the | non-shortcut |     | classes. | A higher | threshold |     | corre- |
| quency patches |                                       | from one   | of the | top-N  | frequency | subsets     |         |              |     |          |          |           |     |        |
spondstostrongershortcuts,asdominantfrequenciesalone
in the previous stage, with a smaller patch size. This con- are sufficient for accurate classification. An example of
| fines the | search | space. | In the final | search | stage, | the top-1 |     |     |     |     |     |     |     |     |
| --------- | ------ | ------ | ------------ | ------ | ------ | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
theresultsofthesemetricsatdifferentthresholdvaluesare
frequencysubsetofeachclassisconsidereddominatingthe
|     |     |     |     |     |     |     | given in | Fig. | 3, where | the | size of | each point | reflects | the |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---- | -------- | --- | ------- | ---------- | -------- | --- |
classificationoftheclassconcerned. Sameas[39],weuse numberofclassesinthetwogroups. Largerpointsindicate
| binary masks | to  | represent | these | frequency | subsets, | called |          |        |             |     |                |     |          |       |
| ------------ | --- | --------- | ----- | --------- | -------- | ------ | -------- | ------ | ----------- | --- | -------------- | --- | -------- | ----- |
|              |     |           |       |           |          |        | a higher | number | of classes, |     | which suggests |     | that the | model |
DominantFrequencyMaps(DFMs). hasastrongertendencytoshortcutlearning.
The sampling percentage p% of frequency patches, the WedonotaverageTPRDFM
acrossallclassesbecause
numberofsampledfrequencysubsetsB andthefrequency its value for many non-shortcut classes is close to zero
resolutionofDFMsarehyperparametersofthesearchalgo-
|     |     |     |     |     |     |     | (specifically | for | large | datasets). | This | can result | in  | a close- |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----- | ---------- | ---- | ---------- | --- | -------- |
rithm(detailsareinthesupplementarymaterial).
|     |     |     |     |     |     |     | to-zeroAvgTPRDFM |     |     | whichdoesnotprovideusefulinfor- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | ------------------------------- | --- | --- | --- | --- |
mationforfurtheranalysis.
3.2.Measuringthedegreeofshortcutlearning
3.3.Impactofshortcutongeneralization
Analyzingmodelslearningbehaviorinaclass-wisemanner,
as done in [39], is labor-intensive if there are hundreds or To assess how frequency shortcuts impact the general-
thousandsofclasses.Toprovideabroadoverviewofmodel ization and robustness performance of models trained on
learningbehaviorontrainingdata,wecategorizeallclasses ImageNet-1k, we evaluate their performance on several
inadatasetintotwogroups: (1)classessubjecttoshortcuts datasets. WeperformIDtestsonImageNet-1k(IN-1k)[6]
| and(2)non-shortcutclasses. |     |     | Aclassisconsideredsubject |     |     |     |                 |     |         |      |      |            |     |         |
| -------------------------- | --- | --- | ------------------------- | --- | --- | --- | --------------- | --- | ------- | ---- | ---- | ---------- | --- | ------- |
|                            |     |     |                           |     |     |     | and ImageNet-v2 |     | (IN-v2) | [31] | test | sets. They | are | consid- |
toshortcutsifitstruepositiverate(TPR)surpassesagiven eredin-distributionastheyarecollectedusingthesamepro-
thresholdwhenthemodelistestedonimagesfilteredtore- tocolasthetrainingset. Furthermore, weuseImageNet-C
tainonlydominantfrequenciesoftheclassconcerned. We (IN-C)[13]tobenchmarktherobustnessofmodelsagainst
useTPR,asahighTPRindicatesthatthefrequencysubset appearanceimagecorruptions,e.g. noise,blurandweather
issufficienttoachieveahighclassificationrate,therebyact-
|     |     |     |     |     |     |     | changes. | To evaluate |     | the generalizability |     | of  | models | to im- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | -------------------- | --- | --- | ------ | ------ |
ingasashortcut.Thefrequency-filteredimagesarereferred ageswithdifferentrenditions,weuseImageNet-Renditions
toasDFM-filteredimages. (IN-R) [14] and ImageNet-Sketch (IN-S) [36]. IN-R con-
For each class, we test the model on both the original tainsrenditionslikecartoons,paintings,artandtoysof200
test images and DFM-filtered images, computing TPR for classesfromIN-1k.IN-Shasthesamenumberofclassesas
both. We denote the TPR on the original images of class IN-1k, containing images of hand-drawn sketches. These
c as TPR and the TPR on the DFM-filtered images as datasets contain visual renditions, which serve to evaluate
| i   | ci  |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
TPRDFM. IfTPRDFM>t(wheret∈[0,1]isapredefined howtherelianceontexturecues(correspondingtomostfre-
| ci  |     | ci  |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
threshold),classc i isconsideredsubjecttoshortcuts;other- quency shortcuts) impacts on generalization. To measure
wise,itisconsideredasoneofthenon-shortcutclasses. We theadversarialrobustnessofmodels,weapplythefastgra-
compute the average TPR values of both groups, shortcut dientsignmethod(FGSM)attacks[10]tothevalidationset
andnon-shortcutclasses,atdifferentthresholdst. Wenote ofIN-1k,withL =4/255.WecalculatetheaverageTPR
∞
them as AvgTPR @t, AvgTPRDFM@t for the shortcut ofshortcutandnon-shortcutclassesonOODdata, thatwe
|     |     | sct |     | sct |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
AvgTPRDFM
classes, and AvgTPR non−sct @t and @t comparetotheresultsonIDdata.
non−sct
25201

| 4.Experiments |     |     |     |     |     |     |     |     |     | Stage 1 |     |     | Stage 2 |     |
| ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | ------- | --- |
2.13.50
| Westartwithinvestigatingthetrade-offbetweenefficiency |     |               |       |                   |     |       |     | 2.30 |     |     |     |     |     |     |
| ----------------------------------------------------- | --- | ------------- | ----- | ----------------- | --- | ----- | --- | ---- | --- | --- | --- | --- | --- | --- |
| (i.e. required                                        |     | computational | time) | and effectiveness |     | (i.e. |     |      |     |     |     |     |     |     |
2.25
| capability | of finding | shortcuts), |     | varying the | configuration |     |     | 0.8 |     |     |     |     |     |     |
| ---------- | ---------- | ----------- | --- | ----------- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2.20
ofthenumberofsampledfrequencysubsetsateachstage,
2.15
| notedasB | (sistheindexofstage). |     |     | Then,weanalyzefre- |     |     |     |                 |     |     |     |     |     |     |
| -------- | --------------------- | --- | --- | ------------------ | --- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- |
|          | s                     |     |     |                    |     |     |     | ssol tsewol ehT |     |     |     |     |     |     |
2.01.06
quencyshortcutlearningofmodelstrainedonIN-1k,relat-
ingittoperformanceresultsunderdifferentOODscenarios. 0 500 1000 0 1000 2000
|     |     |     |     |     |     |     |     |     |     | Stage 3 |     |     | Stage 4 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | ------- | --- |
2.35
0.4
2.30
Setup. WeuseCIFAR-10(C-10)[18]toexplorethecon-
2.25
| figuration                                      | of B, | and configure |     | HFSS with | four | search |     |         |     |     |     |     |     |     |
| ----------------------------------------------- | ----- | ------------- | --- | --------- | ---- | ------ | --- | ------- | --- | --- | --- | --- | --- | --- |
| stages,withpatchsizeof8×8,4×4,2×2and1×1,respec- |       |               |     |           |      |        |     | 2.02.02 |     |     |     |     |     |     |
2.15
tively. Wethenperformlarger-scaleshortcutanalysisusing
| IN-1k: weusesixsearchstages,withpatchsizeof56×56, |     |     |     |                     |     |     |     | 2.10 |      |         |         |          |         |         |
| ------------------------------------------------- | --- | --- | --- | ------------------- | --- | --- | --- | ---- | ---- | ------- | ------- | -------- | ------- | ------- |
| 28×28,14×14,8×8,4×4and2×2.                        |     |     |     | Thepatch(frequency) |     |     |     | 0.0  |      |         |         |          |         |         |
|                                                   |     |     |     |                     |     |     |     |      | 0.00 | 02.2000 | 0.44000 | 00.62500 | 500.800 | 75010.0 |
resolution in the final stage is the same as that in [39]. At No. of sampled frequency subsets
each stage, we sample p=60% frequency patches to form airplane bird deer frog ship
frequency subsets. It results in DFMs containing about automobile cat dog horse truck
|     |     |     |     |     |     |     | Figure | 4.  | The best | class-wise | loss | vs. | the number | of sampled |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | -------- | ---------- | ---- | --- | ---------- | ---------- |
5%frequenciesofthewholespectrumofImageNetimages
andabout15%frequenciesofCIFARimages. Thenumber frequencysubsetsateachstage.
| of training | images | sampled | for | shortcut information |     | eval- | sizedecreases. |     |     |     |     |     |     |     |
| ----------- | ------ | ------- | --- | -------------------- | --- | ----- | -------------- | --- | --- | --- | --- | --- | --- | --- |
uation is the same as that of their corresponding test set. We run the configuration CF-1 five times and note the
Weevaluatethegeneralizationandrobustnessperformance
|     |     |     |     |     |     |     | stabilityofourHFSSsearchalgorithms. |     |     |     |     |     | Wetrackthebest |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | -------------- | --- |
ofImageNet-trainedmodelsonIN-v2[31],IN-C[13],IN- (lowest)lossvaluesachievedforeachclasswhenthemodel
| R [14], | IN-S [36] | and using | FGSM | [10] attacks. |     | We also |     |        |                 |     |         |     |         |              |
| ------- | --------- | --------- | ---- | ------------- | --- | ------- | --- | ------ | --------------- | --- | ------- | --- | ------- | ------------ |
|         |           |           |      |               |     |         | is  | tested | on DFM-filtered |     | images. |     | In Fig. | 4, we report |
carryoutexperimentsonImageNet-10(IN-10)[15]witha the tracking statistics with average loss values (lines) and
ResNet18 model to compare HFSS with single-frequency standard deviations (shadows) over the five trials for each
| removal-based |     | method in | [39]. | Following | their setups, | we  |        |       |     |         |      |               |     |                |
| ------------- | --- | --------- | ----- | --------- | ------------- | --- | ------ | ----- | --- | ------- | ---- | ------------- | --- | -------------- |
|               |     |           |       |           |               |     | class, | which | are | further | used | as a starting |     | point to opti- |
use ImageNet-SCT (IN-SCT) for OOD evaluation. Our mize the configuration of HFSS. As the number of sam-
| configuration | of  | HFSS | enables direct | comparison |     | with the |     |     |     |     |     |     |     |     |
| ------------- | --- | ---- | -------------- | ---------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
pledfrequencysubsetsincreases,thebestlossforeachclass
results reported in [39] as DFMs contain around 5% fre- converges,indicatingthestabilityofHFSSinidentifying
quenciesofwholespectrum. Wereporttrainingconfigura- similar shortcuts learned by models when a sufficient
tionsandadditionalresultsinthesupplementarymaterial.
amountofcandidatefrequencysubsetsaresampledfor
|     |     |     |     |     |     |     | validating |     | shortcut | information. |     | The | standard | deviation |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | -------- | ------------ | --- | --- | -------- | --------- |
4.1.ConfigurationofHFSS recordedatdifferentstagesisrelativelysmallcomparedto
Sampled frequency subsets. At stage s, we sample B the range of best loss. For class airplane we observe an
s
|           |          |      |           |             |     |           | outlier | outcome |     | at stage | 3. In | the supplementary |     | material, |
| --------- | -------- | ---- | --------- | ----------- | --- | --------- | ------- | ------- | --- | -------- | ----- | ----------------- | --- | --------- |
| frequency | subsets, | such | that HFSS | can explore |     | different |         |         |     |          |       |                   |     |           |
combinations of frequency patterns, validating their con- wepresentvisualizationofshortcutcues(DFM-filteredim-
|                 |     |              |          |          |     |           | ages) | computed |     | across | the five | trials. | These | visualizations |
| --------------- | --- | ------------ | -------- | -------- | --- | --------- | ----- | -------- | --- | ------ | -------- | ------- | ----- | -------------- |
| tained shortcut |     | information. | However, | sampling |     | an exces- |       |          |     |        |          |         |       |                |
showthattheidentifiedshortcutpatternsdifferintheorien-
sivenumberofsubsetsincreasestherequiredsearchtime,as
moresubsetsareforevaluatingtheircontributionstoclassi- tationsofstrip-likefeatures,correspondingtolinefeatures
intheimageofairplane.Thissuggeststhatfrequencyshort-
ficationresults.Toexploretheimpactofthenumberofsam-
pling operation B on shortcut identification, we perform cutsdonotarisefromfixedfrequencysubsets; rather, spa-
s
tialfeaturestheycorrespondtomaybevisuallysimilarbut
| experiments | on  | C-10 using | a ResNet18 | backbone, |     | setting |          |     |              |     |              |             |     |           |
| ----------- | --- | ---------- | ---------- | --------- | --- | ------- | -------- | --- | ------------ | --- | ------------ | ----------- | --- | --------- |
|             |     |            |            |           |     |         | composed |     | of different |     | frequencies, | determining |     | an higher |
thenumberofsampledfrequencysubsetsineachstagetobe
as large as possible under limited computational time. We standarddeviationinFig.4.
hypothesizethatHFSSobtainsstableresultsgivenenough
candidatefrequencysubsets. WesetB ,B ,B ,andB to Efficiency vs. effectiveness. Based on the results
|     |     |     |     | 1   | 2 3 | 4   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
be1000,2000,4000and8000respectively,whichwenote inFig.4,weoptimizeCF-1forefficiencyandeffectiveness
asconfigurationCF-1. Theincreasingnumberofcandidate in shortcut identification. The required search time scales
frequencysubsetsatconsecutivestagesisdeterminedbythe upproportionallytoB ,i.e. themorecandidatefrequency
s
increaseinpossiblefrequencypatchcombinationsaspatch subsetsaresampled,themoretimeisneededtoverifytheir
25202

shortcutsidentifiedbyusingthemorecomplexCF-1atlow
0.8
|     |     |     |     |     |     |     | threshold | level. | Although | CF-2.10 | misses | stronger |     | short- |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | -------- | ------- | ------ | -------- | --- | ------ |
0.7 cuts (at higher threshold levels), it achieves a ∼200× re-
|     |     | RPT |     |     |     |     | duction | in computational |     | time | compared | to  | CF-1. | A more |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------------- | --- | ---- | -------- | --- | ----- | ------ |
0.6
time-efficientconfigurationislesseffectiveatfindingshort-
0.5
cutsthanthemorecomplexone,butitstillmanagestoiden-
|     |     | 0.4 |     |     |     |     | tifyshortcutsatlowthresholds. |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- |
102 103 104 The observations made on C-10 allow to estimate ini-
No. of sampled frequency subsets (log scale)
|        |            |     |            |             |        |          | tialconfigurationsforanalysisonlarger-scaledatasets. |     |     |     |     |     |     | We  |
| ------ | ---------- | --- | ---------- | ----------- | ------ | -------- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| Figure | 5. Average | TPR | vs. search | time, where | search | time in- |                                                      |     |     |     |     |     |     |     |
trackthelossstatisticsofHFSSappliedtoImageNet(pro-
creasesproportionallywiththenumberofsampledfrequencysub-
| sets. |     |     |     |     |     |     | videdinthe | supplementarymaterial)andchoosethe |     |     |     |     |     | value |
| ----- | --- | --- | --- | --- | --- | --- | ---------- | ---------------------------------- | --- | --- | --- | --- | --- | ----- |
B forImageNetexperimentsinSec.4.2astheonewhere
|     |     | 10  |     |     |     |     | s   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ssalc tuctrohs fo .oN the loss values show no significant decrease. Given the
8
|     |     |     |     |     |     |     | computational                                     |     | intensity | of identifying |     | shortcuts | in  | large- |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --------- | -------------- | --- | --------- | --- | ------ |
|     |     | 6   |     |     |     |     | scaledatasets,ourworkonIN-1kprimarilyaimsatuncov- |     |           |                |     |           |     |        |
|     |     | 4   |     |     |     |     | eringshortcutswithaffordablecomputationalefforts. |     |           |                |     |           |     |        |
2
|     |     | CF. 1 |     |     |     |     | 4.2.ShortcutidentificationonIN-1k |     |     |     |     |     |     |     |
| --- | --- | ----- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- |
CF. 2.10
0
WeapplyHFSStoexaminefrequencyshortcutslearnedby
|     |     | 0.2 | 0.4 | 0.6 0.8 |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t
Figure6. Thenumberofshortcutclassesgivendifferentthresh- ImageNet models and how they impact model generaliza-
|                             |     |     |     |                          |     |     | tion and | robustness. | The | search | configurations |     | for | Ima- |
| --------------------------- | --- | --- | --- | ------------------------ | --- | --- | -------- | ----------- | --- | ------ | -------------- | --- | --- | ---- |
| oldtsearchbyCF-1andCF-2.10. |     |     |     | UsingCF-2.10uncoversmost |     |     |          |             |     |        |                |     |     |      |
shortcutsidentifiedbyusingCF-1atlowthresholds,withcompu- geNet models, aimed at time-efficiency and effectiveness
tationaltimereducedtoafactorofaround200. atfindingshortcuts,areinthesupplementarymaterial.
| relevance | to  | classification. | We  | perform experiments |     | using |         |           |       |     |       |     |         |        |
| --------- | --- | --------------- | --- | ------------------- | --- | ----- | ------- | --------- | ----- | --- | ----- | --- | ------- | ------ |
|           |     |                 |     |                     |     |       | Results | on IN-1k, | IN-v2 | and | IN-C. | In  | Figs. 7 | and 8, |
differentB
s ,investigatinghowthesizeofthesearchspace we report the results of the analysis of impact of short-
affectstheidentificationofshortcuts. Basedonthetracking cuts on model performance. In general, models perform
| statisticsinFig.4,whichshowsthatincreasingB |     |     |     |     |     | beyonda |                                                      |     |     |     |     |     |     |     |
| ------------------------------------------- | --- | --- | --- | --- | --- | ------- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|                                             |     |     |     |     |     | s       | betterpredictionsonimagesofclassesaffectedbyshortcut |     |     |     |     |     |     |     |
certainpointdoesnotdecreaselosssignificantly,weadjust learning (blue lines) compared to non-shortcut classes (or-
| B to200,B |      | to800andB       | to4000(CF-2.1). |                   |     | Wefurther |                                       |           |     |            |             |               |     |         |
| --------- | ---- | --------------- | --------------- | ----------------- | --- | --------- | ------------------------------------- | --------- | --- | ---------- | ----------- | ------------- | --- | ------- |
| 1         |      | 2               | 4               |                   |     |           | angelines)inbothIDandcorruptiontests. |           |     |            |             | Modelssubject |     |         |
| reduceB   | andB | to2000(CF-2.2). |                 | Wealsodesigneight |     |           |                                       |           |     |            |             |               |     |         |
|           | 3    | 4               |                 |                   |     |           | to frequency                          | shortcuts |     | yield good | performance |               | in  | the ro- |
other configurations (see details in the supplementary ma- bustnessandgeneralizationtestswhentexturesinformation
| terial), | with | CF-2.10 being | the | most efficient | as  | it samples |            |            |     |        |     |     |         |      |
| -------- | ---- | ------------- | --- | -------------- | --- | ---------- | ---------- | ---------- | --- | ------ | --- | --- | ------- | ---- |
|          |      |               |     |                |     |            | is largely | preserved. | For | ResNet | and | ViT | models, | when |
thelowestnumberofcandidatefrequencysubsets. t≥0.8, the AvgTPRDFM exceeds AvgTPR , meaning
|     |         |                 |     |         |       |         |     |     | sct |     |     |     | sct |     |
| --- | ------- | --------------- | --- | ------- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
| To  | compare | the performance |     | of HFSS | under | differ- |     |     |     |     |     |     |     |     |
thatthesemodelsrelypredominantlyondominantfrequen-
ent configurations, we compute the average TPR over all cies(correspondingtosimplefeatures)forclassificationof
classeswhentestedonDFM-filteredimages. Thisindicates shortcut-affected classes, preventing from learning class-
| the general |     | relevance of | the searched | frequency |     | subsets to |         |           |           |     |           |     |        |      |
| ----------- | --- | ------------ | ------------ | --------- | --- | ---------- | ------- | --------- | --------- | --- | --------- | --- | ------ | ---- |
|             |     |              |              |           |     |            | related | semantics | features. | For | instance, | at  | t=0.9, | only |
classification. A higher average TPR indicate higher rele- one class (‘window screen’) is found subject to frequency
| vance, | thus | the frequency | subsets | contain | more | frequency |            |          |          |     |          |     |               |     |
| ------ | ---- | ------------- | ------- | ------- | ---- | --------- | ---------- | -------- | -------- | --- | -------- | --- | ------------- | --- |
|        |      |               |         |         |      |           | shortcuts: | ResNet18 | achieves |     | TPR=0.66 | on  | full-spectrum |     |
shortcutinformation. Weruneachconfigurationfivetimes, images of this class and TPR=0.94 on DFM-filtered im-
computing the mean and standard deviations of average ages (similarly to ResNet50 and ViT-b). Differently, CCT
TPRs over the five trials. We report the results in Fig. 5 AvgTPRDFM@0.9
|     |     |     |     |     |     |     | exhibits | lower |     |     | than | AvgTPR | sct | @0.9, |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | --- | --- | ---- | ------ | --- | ----- |
sct
where the left-most point corresponds to average TPR of indicatingthatwhilefrequencyshortcutscontributesignif-
CF-2.10andtheright-mostpointcorrespondstothatofCF-
|     |     |     |     |     |     |     | icantly to | classification, |     | this model |     | also manage | to  | lever- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --------------- | --- | ---------- | --- | ----------- | --- | ------ |
1. Weobservethatasmorefrequencysubsetsaresampled, age other semantic information: a strong frequency short-
average TPR increases and saturates. This demonstrates cut (e.g. when t>0.7) does not necessarily mean that it is
| that | sampling | more candidate |     | frequency | subsets | does not |          |             |     |       |          |                 |     |     |
| ---- | -------- | -------------- | --- | --------- | ------- | -------- | -------- | ----------- | --- | ----- | -------- | --------------- | --- | --- |
|      |          |                |     |           |         |          | the only | information | a   | model | uses for | classification. |     | We  |
significantly improve the performance of HFSS, while re- furthercompareCCTwithResNet50astheyperformsim-
quiringmorecomputations. ilarly on IN-1k (80.57% and 80.1% respectively). In IN-
We compare CF-1 and CF-2.10 in Fig. 6 based on the v2andIN-C,CCTachieves74.81%and57.73%prediction
number of classes with identified shortcuts at different rates,havingbetterperformancethanResNet50(74.17%in
thresholds. WeobservethatCF-2.10uncoversmostofthe IN-v2 and 48.85% in IN-C). Considering the highest de-
25203

|     |     | IN-1k | IN-v2 |     | IN-C |     |     | IN-S |     | FGSM |     |     |
| --- | --- | ----- | ----- | --- | ---- | --- | --- | ---- | --- | ---- | --- | --- |
1.0
1.0
81teNseR
0.5
0.8
0.0
1.0
05teNseR
0.5
0.6
RPTgvA
0.0
1.0
|     | TCC 0.4 |     |     |     |     |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.5
0.0
1.0
0.2
b-TiV
0.5
0.0 0.0
0.0 0.25 0.50 0.75 0.2 0.25 0.50 0.75 0.4 0.25 0.50 0.75 0.6 0.25 0.50 0.75 0.8 0.25 0.50 0.75 1.0
t
|     |     |     | AvgTPRsct | AvgTPRnon |     | AvgTPRD | F M  | AvgTPRD F | M     |     |     |     |
| --- | --- | --- | --------- | --------- | --- | ------- | ---- | --------- | ----- | --- | --- | --- |
|     |     |     |           |           | sct |         | sc t | no        | n sct |     |     |     |
Figure7. AverageTPRofshortcutandnon-shortcutclassesgivendifferentthresholdsondatasetswith1000classes. Modelsgenerally
perform better on images of shortcut classes than non-shortcut classes. This does not hold for results on IN-S, which lacks preserved
textureinformation.
tobenefitmodelrobustnessandgeneralizationperformance
|     |     | IN-200 | IN-R |     |     |     |     |     |     |     |     |     |
| --- | --- | ------ | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1.0
1.0 understatisticaldistributionshiftsandcorruptionscenarios.
81teNseR
|     | 0.5 |     |     |     |     | Results                                            | on IN-S | and IN-R. | Different |     | from | the results |
| --- | --- | --- | --- | --- | --- | -------------------------------------------------- | ------- | --------- | --------- | --- | ---- | ----------- |
|     | 0.8 |     |     |     |     | onIN-v2andIN-C,frequencyshortcutsimpairgeneraliza- |         |           |           |     |      |             |
0.0
|     |     |     |     |     |     | tion of | models | to texture | and rendition |     | changes. | Model |
| --- | --- | --- | --- | --- | --- | ------- | ------ | ---------- | ------------- | --- | -------- | ----- |
05teNseR 1.0
|     |     |     |     |     |     | performance | on  | shortcut | and non-shortcut |     | classes | on IN- |
| --- | --- | --- | --- | --- | --- | ----------- | --- | -------- | ---------------- | --- | ------- | ------ |
0.5 0.6 S is close to each other. In Fig. 8, we observe that short-
|     | RPTgvA |     |     |     |     | cutclasseshaveworsepredictionresultsthannon-shortcut |     |     |     |     |     |     |
| --- | ------ | --- | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
0.0
|     | 1.0 |     |     |     |     | classes           | when | tested on IN-R                        | which | contains | images | with |
| --- | --- | --- | --- | --- | --- | ----------------- | ---- | ------------------------------------- | ----- | -------- | ------ | ---- |
|     |     |     |     |     |     | renditionchanges. |      | Thisisattributabletothefactthatshort- |       |          |        |      |
TCC 0.4
0.5 cut information are not available in the OOD tests, as the
renditionandsketchtestsetspreservelessorverydifferent
0.0
textureinformationthanIN-CandIN-v2(seegreenlinesof
1.0 0.2
|     | b-TiV |     |     |     |     | IN-SandIN-1k,IN-RandIN-200inFig.7andFig.8).Fre-      |     |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|     | 0.5   |     |     |     |     | quencyshortcutlearningisanexplicitcauseofthetexture- |     |     |     |     |     |     |
biasofImageNetmodels,andtheresultingimpairedgener-
0.0 0.0
0.0 0.25 0.2 0.50 0.75 0.4 00..625 0.500.8 0.75 1.0 alizabilitytorenditionsisinlinewith[8].
t
AvgTPRsct AvgTPRD F M ResultsonFGSMattacks. AsshowninFig.7, thefour
sc t
AvgTPRnon sct AvgTPRD F M models,underadversarialattacks,achievehigherAvgTPR
no n sct
Figure 8. Average TPR of shortcut and non-shortcut classes at forshortcutclasses(especiallythosewithstrongshortcuts)
differentthresholdsondatasetswith200classes.Modelsperform thannon-shortcutclasses. Thissuggeststhatmodelscanbe
betteronshortcutclassesthannon-shortcutclassesinIN-200.IN-
|         |           |                     |              |             |     | inherently                        | robust | to adversarial | noise | if they               | leverage | fre- |
| ------- | --------- | ------------------- | ------------ | ----------- | --- | --------------------------------- | ------ | -------------- | ----- | --------------------- | -------- | ---- |
| R lacks | preserved | texture information | and thus the | models have |     |                                   |        |                |       |                       |          |      |
|         |           |                     |              |             |     | quencyshortcutsforclassification. |        |                |       | Thisisnotsurprisingas |          |      |
worseperformanceonshortcutclassesthannon-shortcutclasses.
adversarialnoiseshardlymanipulatethetexturesofimages.
gree of shortcut learning of CCT among the models (indi- CCT,whichshowsthehighestdegreeoffrequencyshortcut
catedbythemarkersize),inducingfrequencyreliancewith- learning among the considered models, achieves the best
outblockingthelearningofothersemanticfeaturesappears adversarial robustness. This indicates that model reliance
25204

|     |     |     |     |     |     |     |     |     | TPRresultsonDFM-filteredIN-10images. |     |     |     |     | TPR≥0.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | --- | ------- |
Table2.
|     |     | IN-1k |     | IN-C | IN-S |     | FGSM |     |     |     |     |     |     |     |
| --- | --- | ----- | --- | ---- | ---- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
(astrongfrequencyshortcut)ishighlightedinbold.
 (Contrast)
Class
la airlinerwagon h u m - S ia m- ox g o l d e n frog zebra C o n - truck
|     | n   |     |     |     |     |     |     | Method |     |        |      |                |     |        |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | ------ | ---- | -------------- | --- | ------ |
|     | ig  |     |     |     |     |     |     |        |     | b ir d | ca t | re t r i e ver |     | sh i p |
irO
|     |             |         |     |         |         |     |         | HFSS                                    | 0.88 | 0 0.52            | 0.98 0.9 | 0.26        | 0.78          | 0.7 0.76 0.5  |
| --- | ----------- | ------- | --- | ------- | ------- | --- | ------- | --------------------------------------- | ---- | ----------------- | -------- | ----------- | ------------- | ------------- |
|     |             |         |     |         |         |     |         | [39]                                    | 0.08 | 0 0.4             | 0.8 0.02 | 0.02        | 0.14          | 0.8 0.54 0.06 |
|     |             | Dugong  |     | Dugong  | Buckler |     | Dugong  |                                         |      |                   |          |             |               |               |
|     | Prediction  | (99.9%) |     | (99.8%) | (9.9%)  |     | (33.2%) |                                         |      |                   |          |             |               |               |
|     |             |         |     |         |         |     |         | Table 3.                                | TPR  | results on IN-SCT |          | (first row) | and           | corresponding |
|     | d           |         |     |         |         |     |         | DFM-filteredimages(secondandthirdrows). |      |                   |          |             | TPRaboveaver- |               |
e
|     | re  |     |     |     |     |     |     | ageTPR(0.367)ishighlightedinbold. |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- |
tli
-f
M
F
|     | D   |     |     |     |     |     |     | Class |      |                   |          |      |      |                    |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ---- | ----------------- | -------- | ---- | ---- | ------------------ |
|     |     |     |     |     |     |     |     |       | Mil- | car lorikeettabby | holstein | Lab- | tree | horse fishing fire |
Dugong Dugong Roundworms Dugong Method aircraft cat retrieverfrog vessel truck
|     | Prediction  | (82.64) |     | (79.6%) | (21.4%) |     | (14.7%) |      |       |             |             |        |       |                   |
| --- | ----------- | ------- | --- | ------- | ------- | --- | ------- | ---- | ----- | ----------- | ----------- | ------ | ----- | ----------------- |
|     |             |         |     |         |         |     |         | —    | 0.343 | 0.43 0.443  | 0.271 0.329 | 0.3857 | 0.4   | 0.029 0.429 0.629 |
|     |             |         |     |         |         |     |         | HFSS | 0.514 | 0.029 0.586 | 0.886 0.457 | 0.271  | 0.871 | 0.243 0.543 0.071 |
Figure9.Animageofdugongcontainsfrequencyshortcutswhich [39] 0 0 0.2143 0.1286 0.0429 0.0286 0.0571 0.1286 0.2143 0
| amodelreliesonforhigh-confidenceclassification. |     |     |     |     |     |     | Itspresence |     |     |     |     |     |     |     |
| ----------------------------------------------- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
inOODtestsetslikeIN-C,andintheimageunderFGSMattacks thus validate the impact of shortcuts searched by HFSS in
| yields | correct | predictions. |     | But the | reliance | on shortcuts | impairs |                |     |                     |     |             |     |            |
| ------ | ------- | ------------ | --- | ------- | -------- | ------------ | ------- | -------------- | --- | ------------------- | --- | ----------- | --- | ---------- |
|        |         |              |     |         |          |              |         | generalization |     | tests, by measuring |     | performance |     | results on |
modelgeneralizationperformancetorenditionschanges.
|     |     |     |     |     |     |     |     | IN-SCT, | an OOD | test set | designed | by  | [39]. | We report the |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------ | -------- | -------- | --- | ----- | ------------- |
airliner,
|     |     |     |     |     |     |     |     | results | in Tab. | 3. The strong | shortcuts |     | for classes |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- | ------------- | --------- | --- | ----------- | --- |
onfrequencyshortcutscanbenefittheiradversarialrobust-
frogandcontainershipsearchedbyHFSSarepresentinthe
ness,asthesimpleshortcutfeaturesarerobusttoadversarial
OODset,contributingtoclose-tooraboveaverageTPRof
noise.
|     |     |     |     |     |     |     |     | classes | Military | aircraft | (Mil-aircraft), |     | tree | frog, and fish- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | -------- | --------------- | --- | ---- | --------------- |
Summary. Whetherfrequencyshortcutsimpairorbenefit ing vessel in the IN-SCT dataset. The shortcuts searched
|       |                |     |     |            |             |     |         | by [39] | for class | Siamese | cat result | in  | a much | lower TPR |
| ----- | -------------- | --- | --- | ---------- | ----------- | --- | ------- | ------- | --------- | ------- | ---------- | --- | ------ | --------- |
| model | generalization |     | and | robustness | performance |     | depends |         |           |         |            |     |        |           |
onthespecificOODscenarios.InFig.9,weshowanexam- forclasstabbycatinIN-SCT,thoughshortcutsforthisclass
|     |       | dugong |     |       |             |     |               | stillexistintheOODdata(asobservedfromtheresultsof |     |     |     |     |     |     |
| --- | ----- | ------ | --- | ----- | ----------- | --- | ------------- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| ple | image | of     | and | model | predictions |     | in OOD tests. |                                                   |     |     |     |     |     |     |
Thepreservedtextureinformationyieldscorrectpredictions HFSS).Thisindicatesthatthemethodproposedin[39]has
on IN-C and under FGSM attacks, but it is not helpful for limitations in estimating the impact of shortcuts on OOD
|       |                  |     |     |              |          |     |              | performance, |     | andfurtherdemonstratestheeffectivenessof |     |     |     |     |
| ----- | ---------------- | --- | --- | ------------ | -------- | --- | ------------ | ------------ | --- | ---------------------------------------- | --- | --- | --- | --- |
| model | generalizability |     |     | to rendition | changes. |     | The findings |              |     |                                          |     |     |     |     |
reveal a limitation in the design of current OOD bench- HFSSinfindingshortcuts.
| marks, | which            | overlooks |     | the impact | of      | frequency | shortcuts   |     |     |     |     |     |     |     |
| ------ | ---------------- | --------- | --- | ---------- | ------- | --------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
| on     | generalizability |           | of  | models     | and its | relation  | to specific |     |     |     |     |     |     |     |
5.Conclusions
| characteristicsofOODdata. |     |     |     | UsingHFSSforperformance |     |     |     |     |     |     |     |     |     |     |
| ------------------------- | --- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
evaluationcanbridgesuchgap. We proposed the first method for analysis of frequency
|     |     |     |     |     |     |     |     | shortcuts, | HFSS, | that enables |     | the inspection |     | of shortcut |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ----- | ------------ | --- | -------------- | --- | ----------- |
4.3.Comparisonwiththeexistingapproach
learninginlarge-scalemodelsanddatasetswiththousands
Moreeffectiveatfindingshortcuts. Weperformexper- of classes. HFSS is more time-efficient and effective
iments on IN-10 for a direct comparison with the only ex- in finding frequency shortcuts compared to existing ap-
isting method for frequency shortcut analysis [39], which proaches. We investigate frequency shortcut learning in
| evaluates |     | the contribution |     | of single | frequencies |     | to short- |        |         |             |     |        |       |                |
| --------- | --- | ---------------- | --- | --------- | ----------- | --- | --------- | ------ | ------- | ----------- | --- | ------ | ----- | -------------- |
|           |     |                  |     |           |             |     |           | models | trained | on ImageNet | and | relate | their | results to ro- |
cuts and is limited to analyzing models trained on small- bustness and generalization performance under different
scaledatasets. WereportresultsinTab.2. HFSSuncovers OOD conditions. The impact of frequency shortcuts on
strongfrequencyshortcutsforclassesairliner,Siamesecat modelgeneralizationdependsonthespecificOODscenar-
(Siam-cat), ox, frog, zebra, and container ship (Con-ship), ios.Existingmodelsyieldgoodperformanceongeneraliza-
| while | [39] | was | only able | to find | strong | frequency | short- |                 |     |              |     |              |     |               |
| ----- | ---- | --- | --------- | ------- | ------ | --------- | ------ | --------------- | --- | ------------ | --- | ------------ | --- | ------------- |
|       |      |     |           |         |        |           |        | tion benchmarks |     | when texture |     | information, |     | corresponding |
cuts for classes zebra and Siamese cat. The better effec- to most frequency shortcuts, is mostly preserved in OOD
tiveness of HFSS at finding shortcuts is attributable to the data. Instead,frequencyshortcutsimpairthegeneralizabil-
factthatHFSSconsidersthejointcontributionsoffrequen- ityofmodelstoimageswithrenditionchanges. Thishigh-
ciestoclassification,whilein[39]therelevanceofasingle lights the limitation of current OOD performance evalua-
frequencywasmeasurediteratively. tionbenchmarks,whichneedtoexplicitlytakeintoaccount
Frequency shortcuts can impair the generalization per- the impact that frequency shortcuts have on model perfor-
formance of models or provide a false impression of good mance. HFSSprovidesatooltobridgethisgapandextends
generalizationwhenshortcutsexistinOODtests[39]. We therigorofmodelgeneralizationevaluation.
25205

Acknowledgment [13] DanHendrycksandThomasDietterich. Benchmarkingneu-
ralnetworkrobustnesstocommoncorruptionsandperturba-
ShunxinWangandNicolaStrisciuglioarepartlysupported tions. InInternationalConferenceonLearningRepresenta-
| bytheERJUproject. | Europe’sRailJointUndertakingisa |     |     |     |     |             |     |       |     |     |     |     |     |
| ----------------- | ------------------------------- | --- | --- | --- | --- | ----------- | --- | ----- | --- | --- | --- | --- | --- |
|                   |                                 |     |     |     |     | tions,2019. |     | 2,4,5 |     |     |     |     |     |
Europeanpartnershiponrailresearchandinnovationestab-
|     |     |     |     |     |     | [14] DanHendrycks, |     | StevenBasart, |     | NormanMu, |     | SauravKada- |     |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | ------------- | --- | --------- | --- | ----------- | --- |
lishedundertheHorizonEuropeprogram(2020-2027). vath,FrankWang,EvanDorundo,RahulDesai,TylerZhu,
SamyakParajuli,MikeGuo,DawnSong,JacobSteinhardt,
References andJustinGilmer. Themanyfacesofrobustness: Acritical
analysisofout-of-distributiongeneralization.InProceedings
[1] HyojinBahng,SanghyukChun,SangdooYun,JaegulChoo,
oftheIEEE/CVFInternationalConferenceonComputerVi-
andSeongJoonOh.Learningde-biasedrepresentationswith
|     |     |     |     |     |     | sion(ICCV),pages8340–8349,2021. |     |     |     |     | 2,4,5 |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- | ----- | --- | --- |
biased representations. In Proceedings of the 37th Inter- [15] HanxunHuang,XingjunMa,SarahMonazamErfani,James
nationalConferenceonMachineLearning,pages528–539. Bailey, and Yisen Wang. Unlearnable examples: Making
| PMLR,2020.        | 2       |                               |          |          |               |                               |      |           |        |                             |      |         |       |
| ----------------- | ------- | ----------------------------- | -------- | -------- | ------------- | ----------------------------- | ---- | --------- | ------ | --------------------------- | ---- | ------- | ----- |
|                   |         |                               |          |          |               | personaldataunexploitable.    |      |           |        | InInternationalConferenceon |      |         |       |
| [2] Christopher   | Boland, | Sonia                         | Dahdouh, | Sotirios | A. Tsaftaris, |                               |      |           |        |                             |      |         |       |
|                   |         |                               |          |          |               | LearningRepresentations,2021. |      |           |        | 5                           |      |         |       |
| andKeithAGoatman. |         | Therearenoshortcutstoanywhere |          |          |               |                               |      |           |        |                             |      |         |       |
|                   |         |                               |          |          |               | [16] Zhi-Qin                  | John | Xu, Yaoyu | Zhang, | Tao                         | Luo, | Yanyang | Xiao, |
worthgoing: Identifyingshortcutsindeeplearningmodels andZhengMa. Frequencyprinciple: Fourieranalysissheds
formedicalimageanalysis.InSubmittedtoMedicalImaging lightondeepneuralnetworks. CommunicationsinCompu-
| withDeepLearning,2024. |         | underreview. |     | 2         |           |                                       |       |             |      |        |         |     |      |
| ---------------------- | ------- | ------------ | --- | --------- | --------- | ------------------------------------- | ----- | ----------- | ---- | ------ | ------- | --- | ---- |
|                        |         |              |     |           |           | tationalPhysics,28(5):1746–1767,2020. |       |             |      |        | 2       |     |      |
| [3] Yuan Cao,          | Zhiying | Fang, Yue    | Wu, | Ding-Xuan | Zhou, and |                                       |       |             |      |        |         |     |      |
|                        |         |              |     |           |           | [17] Og˘uzhan                         | Fatih | Kar, Teresa | Yeo, | Andrei | Atanov, | and | Amir |
Quanquan Gu. Towards understanding the spectral bias of Zamir. 3dcommoncorruptionsanddataaugmentation. In
deeplearning,2020. 2 ProceedingsoftheIEEE/CVFConferenceonComputerVi-
[4] Francesco Croce, Maksym Andriushchenko, Vikash Se- sionandPatternRecognition,pages18963–18974,2022. 2
| hwag,   | Edoardo Debenedetti, |     | Nicolas  | Flammarion, | Mung         |                     |     |          |          |     |           |          |      |
| ------- | -------------------- | --- | -------- | ----------- | ------------ | ------------------- | --- | -------- | -------- | --- | --------- | -------- | ---- |
|         |                      |     |          |             |              | [18] A. Krizhevsky. |     | Learning | multiple |     | layers of | features | from |
| Chiang, | Prateek Mittal,      | and | Matthias | Hein.       | Robustbench: |                     |     |          |          |     |           |          |      |
|         |                      |     |          |             |              | tinyimages,1970.    |     | 5        |          |     |           |          |      |
astandardizedadversarialrobustnessbenchmark. InThirty- [19] S. Lapuschkin, S. Wa¨ldchen, A. Binder, et al. Unmasking
fifthConferenceonNeuralInformationProcessingSystems CleverHanspredictorsandassessingwhatmachinesreally
DatasetsandBenchmarksTrack(Round2),2021. 2 NatCommun10,1096,2019.
|     |     |     |     |     |     | learn. |     |     |     |     | 1,2 |     |     |
| --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
[5] NikolayDagaev,BrettD.Roads,XiaoliangLuo,DanielN.
|     |     |     |     |     |     | [20] Mathias | Lecuyer, | Vaggelis |     | Atlidakis, | Roxana | Geambasu, |     |
| --- | --- | --- | --- | --- | --- | ------------ | -------- | -------- | --- | ---------- | ------ | --------- | --- |
Barry,KaustubhR.Patil,andBradleyC.Love. Atoo-good- Daniel Hsu, and Suman Jana. Certified robustness to ad-
to-be-truepriortoreduceshortcutreliance,2021. 1,2 versarialexampleswithdifferentialprivacy. In2019IEEE
[6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Symposium on Security and Privacy (SP), pages 656–672,
| andLiFei-Fei. | Imagenet: | Alarge-scalehierarchicalimage |     |     |     |       |     |     |     |     |     |     |     |
| ------------- | --------- | ----------------------------- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
|               |           |                               |     |     |     | 2019. | 2   |     |     |     |     |     |     |
database.In2009IEEEConferenceonComputerVisionand
|     |     |     |     |     |     | [21] Sungbin | Lim, | Ildoo | Kim, Taesup | Kim, | Chiheon | Kim, | and |
| --- | --- | --- | --- | --- | --- | ------------ | ---- | ----- | ----------- | ---- | ------- | ---- | --- |
PatternRecognition,pages248–255,2009. 4 SungwoongKim. Fastautoaugment,2019. 3
[7] Paul Gavrikov and Janis Keuper. Can biases in imagenet [22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei,
models explain generalization? In Proceedings of the ZhengZhang, StephenLin, andBainingGuo. Swintrans-
| IEEE/CVF | Conference | on  | Computer | Vision | and Pattern |         |              |     |        |             |       |         |      |
| -------- | ---------- | --- | -------- | ------ | ----------- | ------- | ------------ | --- | ------ | ----------- | ----- | ------- | ---- |
|          |            |     |          |        |             | former: | Hierarchical |     | vision | transformer | using | shifted | win- |
Recognition(CVPR),pages22184–22194,2024.
|     |     |     |     |     | 1,2 | dows,2021. |     | 3   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
[8] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, [23] Emanuele Marconato, Stefano Teso, Antonio Vergari, and
MatthiasBethge,FelixA.Wichmann,andWielandBrendel. AndreaPasserini. Notallneuro-symbolicconceptsarecre-
Imagenet-trainedCNNsarebiasedtowardstexture;increas- atedequal: Analysisandmitigationofreasoningshortcuts.
| ingshapebiasimprovesaccuracyandrobustness. |     |     |     |     | InInter- |                   |     |            |     |           |             |     |      |
| ------------------------------------------ | --- | --- | --- | --- | -------- | ----------------- | --- | ---------- | --- | --------- | ----------- | --- | ---- |
|                                            |     |     |     |     |          | In Thirty-seventh |     | Conference |     | on Neural | Information |     | Pro- |
nationalConferenceonLearningRepresentations,2019. 1, cessingSystems,2023. 2
2,7 [24] Matthias Minderer, Olivier Bachem, Neil Houlsby, and
[9] Robert Geirhos, Jo¨rn-Henrik Jacobsen, Claudio Michaelis, Michael Tschannen. Automatic shortcut removal for self-
RichardZemel,WielandBrendel,MatthiasBethge,andFe- supervisedrepresentationlearning,2020. 2
lixA.Wichmann.Shortcutlearningindeepneuralnetworks.
|     |     |     |     |     |     | [25] Eric | Mintun, | Alexander | Kirillov, | and | Saining | Xie. | On in- |
| --- | --- | --- | --- | --- | --- | --------- | ------- | --------- | --------- | --- | ------- | ---- | ------ |
NatureMachineIntelligence,2(11):665–673,2020. 1,2 teractionbetweenaugmentationsandcorruptionsinnatural
[10] IanJ.Goodfellow,JonathonShlens,andChristianSzegedy. corruption robustness. In Advances in Neural Information
Explainingandharnessingadversarialexamples,2015. 4,5 Processing Systems, pages 3571–3583. Curran Associates,
| [11] JindongGuandDanielaOelke. |     |     | Understandingbiasinma- |     |     | Inc.,2021. | 2   |     |     |     |     |     |     |
| ------------------------------ | --- | --- | ---------------------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
chinelearning,2019. 2 [26] Samuel G. Mu¨ller and Frank Hutter. Trivialaugment:
[12] JunGuo,WeiBao,JiakaiWang,YuqingMa,XinghaiGao, Tuning-freeyetstate-of-the-artdataaugmentation,2021. 3
GangXiao,AishanLiu,JianDong,XianglongLiu,andWen- [27] MeikeNauta, RickyWalsh, AdamDubowski, andChristin
jun Wu. A comprehensive evaluation framework for deep Seifert. Uncoveringandcorrectingshortcutlearninginma-
modelrobustness. PatternRecognition, 137:109308, 2023. chinelearningmodelsforskincancerdiagnosis. Diagnos-
| 2   |     |     |     |     |     | tics,12(1),2022. |     | 1,2,3 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | ----- | --- | --- | --- | --- | --- |
25206

[28] Hongjing Niu, Hanting Li, Feng Zhao, and Bin Li. Road- [40] Shunxin Wang, Raymond Veldhuis, Christoph Brune, and
blocksfortemporarilydisablingshortcutsandlearningnew NicolaStrisciuglio. Asurveyontherobustnessofcomputer
knowledge. InAdvancesinNeuralInformationProcessing visionmodelsagainstcommoncorruptions,2023. 2
Systems,pages29064–29075.CurranAssociates,Inc.,2022. [41] Zhi-QinJohnXuandHanxuZhou.Deepfrequencyprinciple
| 2   |     |     |     |     |     |     |     | towardsunderstandingwhydeeperlearningisfaster,2020.2 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- |
[29] IanOsbandandBenjaminVanRoy. Whyisposteriorsam- [42] JiayunZhengandMaggieMakar.Causallymotivatedmulti-
pling better than optimism for reinforcement learning? In InAdvancesinNeural
shortcutidentificationandremoval.
Proceedings of the 34th International Conference on Ma- Information ProcessingSystems, pages12800–12812. Cur-
chineLearning,pages2701–2710.PMLR,2017. 3 ranAssociates,Inc.,2022. 2
[30] MohammadPezeshki,Se´kou-OumarKaba,YoshuaBengio, [43] Xin Zou and Weiwei Liu. On the adversarial robustness
| AaronCourville,DoinaPrecup,andGuillaumeLajoie. |     |     |     |     |     |     | Gra- |                        |                |         | Advances |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- | ---------------------- | -------------- | ------- | -------- |
|                                                |     |     |     |     |     |     |      | of out-of-distribution | generalization | models. | In       |
dientstarvation:Alearningproclivityinneuralnetworks.In in Neural Information Processing Systems, pages 68908–
AdvancesinNeuralInformationProcessingSystems,2021. 68938.CurranAssociates,Inc.,2023. 2
2
| [31] Benjamin |     | Recht, Rebecca | Roelofs, |     | Ludwig | Schmidt, | and |     |     |     |     |
| ------------- | --- | -------------- | -------- | --- | ------ | -------- | --- | --- | --- | --- | --- |
VaishaalShankar.DoImageNetclassifiersgeneralizetoIma-
| geNet? | InProceedingsofthe36thInternationalConference |     |     |     |     |     |     |     |     |     |     |
| ------ | --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
onMachineLearning,pages5389–5400.PMLR,2019.2,4,
5
| [32] Joshua                    | Robinson,   |               | Li Sun, Ke   | Yu,                     | Kayhan      | Batmanghelich, |          |     |     |     |     |
| ------------------------------ | ----------- | ------------- | ------------ | ----------------------- | ----------- | -------------- | -------- | --- | --- | --- | --- |
| Stefanie                       | Jegelka,    | and           | Suvrit       | Sra. Can                | contrastive |                | learning |     |     |     |     |
| avoidshortcutsolutions?,2021.  |             |               |              | 2                       |             |                |          |     |     |     |     |
| [33] Ramprasaath               |             | R. Selvaraju, |              | Michael                 | Cogswell,   | Abhishek       |          |     |     |     |     |
| Das,                           | Ramakrishna |               | Vedantam,    | Devi                    | Parikh,     | and Dhruv      | Ba-      |     |     |     |     |
| tra.                           | Grad-cam:   | Visual        | explanations |                         | from        | deep           | networks |     |     |     |     |
| viagradient-basedlocalization. |             |               |              | In2017IEEEInternational |             |                |          |     |     |     |     |
| Conference                     |             | on Computer   | Vision       | (ICCV),                 |             | pages 618–626, |          |     |     |     |     |
| 2017.                          | 2           |               |              |                         |             |                |          |     |     |     |     |
[34] HarshayShah,KaustavTamuly,AditiRaghunathan,Prateek
| Jain,andPraneethNetrapalli. |     |     |                                   | Thepitfallsofsimplicitybias |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --------------------------------- | --------------------------- | --- | --- | --- | --- | --- | --- | --- |
| inneuralnetworks.           |     |     | InAdvancesinNeuralInformationPro- |                             |     |     |     |     |     |     |     |
cessingSystems,pages9573–9585.CurranAssociates,Inc.,
| 2020.       | 1,2    |        |                   |              |           |             |        |     |     |     |     |
| ----------- | ------ | ------ | ----------------- | ------------ | --------- | ----------- | ------ | --- | --- | --- | --- |
| [35] Damien | Teney, | Armand | Mihai             | Nicolicioiu, |           | Valentin    | Hart-  |     |     |     |     |
| mann,       | and    | Ehsan  | Abbasnejad.       | Neural       | redshift: |             | Random |     |     |     |     |
| networks    | are    | not    | random functions. |              | In        | Proceedings | of     |     |     |     |     |
theIEEE/CVFConferenceonComputerVisionandPattern
| Recognition(CVPR),pages4786–4796,2024.   |                                                 |         |                               |          |           | 2      |           |     |     |     |     |
| ---------------------------------------- | ----------------------------------------------- | ------- | ----------------------------- | -------- | --------- | ------ | --------- | --- | --- | --- | --- |
| [36] Haohan                              | Wang,                                           | Songwei | Ge,                           | Zachary  | Lipton,   | and    | Eric P    |     |     |     |     |
| Xing.                                    | Learningrobustglobalrepresentationsbypenalizing |         |                               |          |           |        |           |     |     |     |     |
| localpredictivepower.                    |                                                 |         | InAdvancesinNeuralInformation |          |           |        |           |     |     |     |     |
| ProcessingSystems,pages10506–10518,2019. |                                                 |         |                               |          |           | 2,4,5  |           |     |     |     |     |
| [37] Shunxin                             | Wang,                                           | Raymond | Veldhuis,                     |          | Christoph | Brune, | and       |     |     |     |     |
| Nicola                                   | Strisciuglio.                                   |         | Frequency                     | shortcut | learning  |        | in neural |     |     |     |     |
networks.InNeurIPS2022WorkshoponDistributionShifts:
| ConnectingMethodsandApplications,2022. |               |           |          |              |     | 1,2           |     |     |     |     |     |
| -------------------------------------- | ------------- | --------- | -------- | ------------ | --- | ------------- | --- | --- | --- | --- | --- |
| [38] Shunxin                           | Wang,         | Christoph | Brune,   | Raymond      |     | Veldhuis,     | and |     |     |     |     |
| Nicola                                 | Strisciuglio. |           | Dfm-x:   | Augmentation |     | by leveraging |     |     |     |     |     |
| prior                                  | knowledge     | of        | shortcut | learning.    | In  | Proceedings   | of  |     |     |     |     |
theIEEE/CVFInternationalConferenceonComputerVision
| (ICCV)Workshops,pages129–138,2023.            |        |          |                                  |            | 2,3        |            |          |     |     |     |     |
| --------------------------------------------- | ------ | -------- | -------------------------------- | ---------- | ---------- | ---------- | -------- | --- | --- | --- | --- |
| [39] Shunxin                                  | Wang,  | Raymond  | Veldhuis,                        |            | Christoph  | Brune,     | and      |     |     |     |     |
| NicolaStrisciuglio.                           |        |          | Whatdoneuralnetworkslearninimage |            |            |            |          |     |     |     |     |
| classification?afrequencyshortcutperspective. |        |          |                                  |            |            | InProceed- |          |     |     |     |     |
| ings                                          | of the | IEEE/CVF | International                    |            | Conference |            | on Com-  |     |     |     |     |
| puter                                         | Vision | (ICCV),  | pages                            | 1433–1442, | 2023.      | 1,         | 2, 3, 4, |     |     |     |     |
5,8
25207