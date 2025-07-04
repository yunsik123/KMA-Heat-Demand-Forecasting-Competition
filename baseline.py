import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_heat = pd.read_csv("data/train_heat.csv",encoding="CP949")
df_call119 = pd.read_csv("data/call119_train.csv",encoding="CP949")
df_cat119 = pd.read_csv("data/cat119_train.csv",encoding="CP949")
df4 = pd.read_csv("data/train_subway21.csv",encoding="CP949")
df5 = pd.read_csv("data/train_subway22.csv",encoding="CP949")
df6 = pd.read_csv("data/train_subway23.csv",encoding="CP949")

#1.역난방 열수요와 날씨 빅데이터를 융합한 열수요 예측 
df_heat.info()#시계열 데이터임 일단 결측치없어보임. #499301

#Unnamed: 0             #                   #id    #int  #drop때릴가능성농후
#train_heat.tm          #시간                  #시간인데 2021년1월1일 01시부터부터 20223 12월 31일 23시까지   #int
#train_heat.branch_id   #지사명                  # A~S까지 26279개로  #object
#train_heat.ta          #기온                 #-99부분을 제외한다면 나머지는 그냥 적당히 -25~50사이
#train_heat.wd          #풍량
#train_heat.ws          #풍속
#train_heat.rn_day      #일강수량
#train_heat.rn_hr1      #시간강수량
#train_heat.hm          #상대습도
#train_heat.si          #열사량
#train_heat.ta_chi	    #체감온도
#train_heat.heat_demand #열수요  

df_heat.describe()       #최솟값에 -99존재

df_heat.info()
df_heat['train_heat.branch_id'].value_counts()
df_heat[df_heat['train_heat.branch_id']=='A'] #지사별특징을 알아볼 필요가 있음.


sns.kdeplot(df_heat['train_heat.ta'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_heat['train_heat.wd'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_heat['train_heat.ws'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_heat['train_heat.rn_day'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_heat['train_heat.rn_hr1'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_heat['train_heat.hm'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_heat['train_heat.si'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_heat['train_heat.ta_chi'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_heat['train_heat.heat_demand'], fill=True, bw_adjust=1.5)



#-99를 어떻게 처리할지 고민해봐야함.
#1.걍 -99인것과 아닌것으로 열을 하나만든다.#머신러닝용




#2-1.결측치라고 생각하고 보간한다. #시계열이니까?
df_heat = df_heat.replace(-99, np.nan)
null_counts = df_heat.isna().sum(axis=1)
distribution = null_counts.value_counts().sort_index()
print(distribution) 
#완전한 데이터가 전체의 절반정도밖에 없고 최소1개이상이 null이고 이것저것 쓰가 되어있음 8개의 연속형 변수중 8개가 전부다 null인 애도 있음.
#보간을 잘해야하나? 라는 생각이듬 


#2-2. 열마다의 null개수를 한번 알아보고 싶어짐.
null_per_column = df_heat.isna().sum()
print(null_per_column)

#train_heat.si는 결측치가 절반임. 열을 버리거나 0과1로해야할듯  #난 경험상 버림
#나머지는 적당히 결측치가 존재함. 보간을 하는게 좋을듯? interpolate(method='time')등으로 열심히 보간할듯




##############2.소방 데이터와 날씨 빅데이터를 융합한 119신고 건수 예측

df_call119.info() #42924
df_call119.describe() #여기도 결측치는 -99네?
#Unnamed: 0                              id          int   #2020~2023의 5~10얼
#call119_train.tm                        시간        date       
#call119_train.address_city              시/도명       object    
#call119_train.address_gu                군/구명       object
#call119_train.sub_address               읍/면/동명     object
#call119_train.stn                       AWS지점코드    int
#call119_train.ta_max                     최저기온       float
#call119_train.ta_min                     최고기온       float
#call119_train.ta_max_min                 일교차         float
#call119_train.hm_min                     일강수량        float
#call119_train.hm_max                     최대순간풍속     float
#call119_train.ws_max                      최대풍속        float
#call119_train.ws_ins_max                   최소상대습도    float
#call119_train.rn_day                      최고상대습도    float
#call119_train.call_count                   일신고건수      int


df_call119['call119_train.address_city'].value_counts() #다 부산이네??
df_call119['call119_train.address_gu'].value_counts()   #16개의구
df_call119['call119_train.sub_address'].value_counts()  #136개의 동?
#지도시각화가 필요할지도?

sns.kdeplot(df_call119['call119_train.stn'], fill=True, bw_adjust=1.5)  #위치를 측정한 장소네? 근데 쌍봉분포임 어째서????
df_call119['call119_train.stn'].unique()#[904, 921, 940, 941, 939, 923, 942, 159, 938, 950, 937, 910]
#여기서 데이터 가져와서 join하면 될듯
#https://www.weather.go.kr/w/observation/land/aws-obs.do?db=MINDB_01M&tm=2025.05.27%2011%3A52&stnId=0&sidoCode=2600000000&sort=&config=   

sns.kdeplot(df_call119['call119_train.ta_max'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_call119['call119_train.ta_max_min '], fill=True, bw_adjust=1.5)
sns.kdeplot(df_call119['call119_train.hm_min'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_call119['call119_train.hm_max'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_call119['call119_train.ws_max'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_call119['call119_train.ws_ins_max'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_call119['call119_train.rn_day'], fill=True, bw_adjust=1.5)
sns.kdeplot(df_call119['call119_train.call_count'], fill=True, bw_adjust=1.5)
plt.close('all')

df_call119 = df_call119.replace(-99, np.nan)                                 #보면 결측이 거의없긴하다. 보간하면될듯??/
null_counts = df_call119.isna().sum(axis=1)
distribution = null_counts.value_counts().sort_index()
print(distribution) 


null_per_column2 = df_call119.isna().sum()
print(null_per_column2)
#밑에 두개가 그나마 결측치가 많고 나머지도 조금씩 존재하긴한다.
#call119_train.hm_min          3056
#call119_train.hm_max          3056











df_cat119.info() #9개열 #61771행
df_call119.describe() #여기는 -99인결측치가 안보임???


#cat119_train.tm               id
#cat119_train.tm              시/도명
#cat119_train.address_city      군/구명
#cat119_train.address_gu        읍/면/동명
#cat119_train.sub_address        역명
#cat119_train.cat               신고종별
#cat119_train.sub_cat           신고세부종별
#cat119_train.stn                AWS지점코드
#cat119_train.call_count         일신고건수



#지도시각화와관련하여~~















#1.두데이터에 대한 이해 및 join관련  시/도명,군/구명,읍/면/동명으로 join해도 되고 AWS지점으로해도됨
#2.시각화
#3.관련 추가데이터
#4.추가논문


#3번관련 더 찾아봐야할 것 찾기