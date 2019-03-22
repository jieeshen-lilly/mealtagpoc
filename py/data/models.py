import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import json
import boto3
import io
from pytz import timezone
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

conn = boto3.client('s3')
s3 = boto3.resource('s3')
baseline_date = parse('2019-01-01 00:00:00 +0000')
sample_rate = 50

def get_mealtag_catalog(bucketname):
    motion = pd.DataFrame(columns=['user','secelapsed','date','time','timezone','minutes','path'])
    event = pd.DataFrame(columns=['user','secelapsed', 'date','time','timezone','path'])
    missing = pd.DataFrame(columns=['user','secelapsed','date','time','timezone','path'])
    for item in conn.list_objects(Bucket=bucketname)['Contents']:
        filename = item['Key']
        if 'motion' in filename:
            userid, datatype, timestamp = filename.split('.')[0].split('_')
            dt, tm, tzone = timestamp.split(' ')
            x = {
                "user": userid,
                "secelapsed": (parse(timestamp) - baseline_date).total_seconds(),
                "date": dt,
                "time": tm,
                "timezone":tzone,
                "minutes": int(item['Size'] * 0.0105 / 3000),
                "path":f"s3://{bucketname}/{filename}"
            }
            motion = motion.append(x, ignore_index=True)

        if 'event' in filename:
            userid, datatype, timestamp = filename.split('.')[0].split('_')
            dt, tm, tzone = timestamp.split(' ')
            x = {
                "user": userid,
                "secelapsed": (parse(timestamp) - baseline_date).total_seconds(),
                "date": dt,
                "time": tm,
                "timezone":tzone,
                "path":f"s3://{bucketname}/{filename}"
            }
            event = event.append(x, ignore_index=True)
        if 'miss' in filename:
            userid, datatype, timestamp = filename.split('.')[0].split('_')
            dt, tm, tzone = timestamp.split(' ')
            x = {
                "user": userid,
                "secelapsed": (parse(timestamp) - baseline_date).total_seconds(),
                "date": dt,
                "time": tm,
                "timezone":tzone,
                "path":f"s3://{bucketname}/{filename}"
            }
            missing = missing.append(x, ignore_index=True)

    motion['date'] = motion['date'].astype('datetime64[ns]')
    event['date'] = event['date'].astype('datetime64[ns]')
    missing['date'] = missing['date'].astype('datetime64[ns]')
    return {'motion': motion,'event':event,'missing':missing}


def summarize_motion(df, start_date="2000-01-01", end_date="2100-01-01"):
    df = df[(df["date"]>=parse(start_date)) & (df["date"]<=parse(end_date))]
    x = {
        "no_user": len(df["user"].unique()),
        "total_session": len(df),
        "total_min":df["minutes"].sum(),
        "ave_min":df["minutes"].mean()
    }
    return json.dumps(x)


def day_gram(userid, features, catalog, dates=None, timezone="America/Indiana/Indianapolis"):
    mt = catalog["motion"]
    mt = mt[mt["user"]==userid]
    et = catalog["event"]
    et = et[et["user"]==userid]
    ms = catalog["missing"]
    ms = ms[ms["user"]==userid]

    output_df = []
    if dates is None:
        dates = mt["date"].unique()
    columns=['row','accl_x','accl_y','accl_z','gyro_x','gyro_y','gyro_z']
    for d in dates:
        d_mt_df = pd.DataFrame(columns=columns+['ts'])
        for session in mt[mt["date"]==d]["path"]:
            userid, datatype, timestamp = session.split('.')[0].split('_')
            motion_df = pd.read_csv(session, header=None, names=columns, sep=",", engine='python')
            motion_df['ts'] = motion_df["row"]/sample_rate+ (parse(timestamp) - baseline_date).total_seconds()
            d_mt_df=d_mt_df.append(motion_df, ignore_index=False).reset_index()
    #     print()
    #     print(len(d_mt_df['ts']))
        fig = plt.figure(figsize=(30,5))
        for f in features:
            plt.plot(d_mt_df['ts'], d_mt_df[f])
        plt.title(str(d).split("T")[0])

        d_et_df = pd.DataFrame()
        for session in et[et["date"]==d]["path"]:
            event_df = pd.read_csv(session, header=None, sep=':\ ', names=["time","event"], engine='python')
            d_et_df = d_et_df.append(event_df)
        if(len(event_df)==0):
            continue
        start = d_et_df[d_et_df["event"]=='meal started'].iloc[:,0]
        end = d_et_df[d_et_df["event"]=='meal completed'].iloc[:,0]
        utensil = d_et_df[d_et_df["event"].str.match('utensil')].iloc[:,1]
        d_et = pd.concat([start.rename('start').reset_index(drop=True),
                          end.rename('end').reset_index(drop=True), 
                          utensil.rename('utensil').reset_index(drop=True)],axis=1)
        d_ms_df = pd.DataFrame()
        for session in ms[ms["date"]==d]["path"]:
            ms_df = pd.read_csv(session, header=None, sep=':\ ', names=["time","event"], engine='python')
            d_ms_df = d_ms_df.append(ms_df)
        if(len(d_ms_df)==0):
            continue
        start = d_ms_df[d_ms_df["event"]=='meal started'].iloc[:,0]
        end = d_ms_df[d_ms_df["event"]=='meal completed'].iloc[:,0]
        d_ms = pd.concat([start.rename('start').reset_index(drop=True),
                          end.rename('end').reset_index(drop=True)],axis=1)
        meal_df = pd.concat([d_et, d_ms], axis=0)

        meal_list=[]
        d_mt_df["tag"] = 0
        for index, row in meal_df.iterrows():
            mstart = (parse(row["start"])-baseline_date).total_seconds()
            mend = (parse(row["end"])-baseline_date).total_seconds()
            x = {
                "start_seconds":mstart,
                "end_seconds":mend,
                "utensil":str(row["utensil"]).split(" ")[-1]
            }
    #         d_mt_df.loc[(d_mt_df['ts']<mstart) | (d_mt_df['ts']>mend), "tag"] = 0
            d_mt_df.loc[(d_mt_df['ts']>=mstart) & (d_mt_df['ts']<=mend), "tag"] = 1

            meal_list.append(x)

    #     plt.plot(d_mt_df['ts'], d_mt_df["accl_y"])
            plt.axvspan(mstart,mend,color="red",alpha=0.5)
        
        
        output_df.append(d_mt_df)
    return output_df
