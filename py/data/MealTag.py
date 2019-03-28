import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
import boto3
import io
from pytz import timezone
import matplotlib.pyplot as plt


class catalog:
    
    def __init__(self, cfg, tzonedict):
        self.cfg = cfg
        conn = boto3.client('s3')
        s3 = boto3.resource('s3')
        self.tzonedict = tzonedict
        self.filelist = conn.list_objects(Bucket=self.cfg['bucketname'])['Contents']
        self.motion, self.event, self.missing = self.getMealTag()
        
        
    def getMealTag(self):
        baseline_date = self.cfg['reference_time']
        tzonedict = self.tzonedict
        motion = pd.DataFrame(columns=['user','secelapsed','timestamp','date','time','timezone','minutes','path'])
        event = pd.DataFrame(columns=['user','secelapsed','timestamp', 'date','time','timezone','path'])
        missing = pd.DataFrame(columns=['user','secelapsed','timestamp','date','time','timezone','path'])
        for item in self.filelist:
            filename = item['Key']
            if 'motion' in filename:
                userid, datatype, timestamp = filename.split('.')[0].split('_')
                if type(tzonedict) == dict:
                    tz = tzonedict[userid]
                else:
                    tz = tzonedict
                localtime = parse(str(timestamp)).astimezone(timezone(tz))
                x = {
                    "user": userid,
                    "secelapsed": (parse(timestamp) - baseline_date).total_seconds(),
                    "timestamp":timestamp,
                    "date": str(localtime.date()),
                    "time": str(localtime.time()),
                    "timezone":str(localtime.tzinfo),
                    "minutes": int(item['Size'] * 0.0105 / 3000),
                    "path":f"s3://{self.cfg['bucketname']}/{filename}"
                }
                motion = motion.append(x, ignore_index=True).sort_values(by=['secelapsed'])

            if 'event' in filename:
                userid, datatype, timestamp = filename.split('.')[0].split('_')
                dt, tm, tzone = timestamp.split(' ')
                if type(tzonedict) == dict:
                    tz = tzonedict[userid]
                else:
                    tz = tzonedict
                localtime = parse(str(timestamp)).astimezone(timezone(tz))
                x = {
                    "user": userid,
                    "secelapsed": (parse(timestamp) - baseline_date).total_seconds(),
                    "timestamp":timestamp,
                    "date": str(localtime.date()),
                    "time": str(localtime.time()),
                    "timezone":str(localtime.tzinfo),
                    "path":f"s3://{self.cfg['bucketname']}/{filename}"
                }
                event = event.append(x, ignore_index=True).sort_values(by=['secelapsed'])
                
            if 'miss' in filename:
                userid, datatype, timestamp = filename.split('.')[0].split('_')
                dt, tm, tzone = timestamp.split(' ')
                if type(tzonedict) == dict:
                    tz = tzonedict[userid]
                else:
                    tz = tzonedict
                localtime = parse(str(timestamp)).astimezone(timezone(tz))
                x = {
                    "user": userid,
                    "secelapsed": (parse(timestamp) - baseline_date).total_seconds(),
                    "timestamp":timestamp,
                    "date": str(localtime.date()),
                    "time": str(localtime.time()),
                    "timezone":str(localtime.tzinfo),
                    "path":f"s3://{self.cfg['bucketname']}/{filename}"
                }
                missing = missing.append(x, ignore_index=True).sort_values(by=['secelapsed'])
        return motion, event, missing

    
class userdata:
    def __init__(self, userid, catalog, cfg, dates=None, tzone="America/Indianapolis"):
        mt = catalog.motion
        self.mt = mt[mt["user"]==userid]
        et = catalog.event
        self.et = et[et["user"]==userid]
        ms = catalog.missing
        self.ms = ms[ms["user"]==userid]
        
        self.cfg = cfg
        self.tzone = tzone
        if (dates is None) or (dates == 'all') :
            self.dates = self.mt["date"].unique()
        
        
    def getMeals(self):
        baseline_date = self.cfg['reference_time']
        meals = {}
        for d in self.dates:
            d_et_df = pd.DataFrame()
            for session in self.et[self.et["date"]==d]["path"]:
                event_df = pd.read_csv(session, header=None, sep=':\ ', names=["time","event"], engine='python')
                d_et_df = d_et_df.append(event_df)

            d_et = pd.DataFrame(columns=['start','end','utensil'])

            if(len(d_et_df)!=0):

                start = d_et_df[d_et_df["event"]=='meal started'].iloc[:,0]
                end = d_et_df[d_et_df["event"]=='meal completed'].iloc[:,0]
                utensil = d_et_df[d_et_df["event"].str.match('utensil')].iloc[:,1]
                d_et = pd.concat([start.rename('start').reset_index(drop=True),
                                  end.rename('end').reset_index(drop=True), 
                                  utensil.rename('utensil').reset_index(drop=True)],axis=1)


            d_ms_df = pd.DataFrame()
            for session in self.ms[self.ms["date"]==d]["path"]:
                ms_df = pd.read_csv(session, header=None, sep=':\ ', names=["time","event"], engine='python')
                d_ms_df = d_ms_df.append(ms_df)

            d_ms = pd.DataFrame(columns=['start','end'])
            if(len(d_ms_df)!=0):
                start = d_ms_df[d_ms_df["event"]=='meal started'].iloc[:,0]
                end = d_ms_df[d_ms_df["event"]=='meal completed'].iloc[:,0]
                d_ms = pd.concat([start.rename('start').reset_index(drop=True),
                                  end.rename('end').reset_index(drop=True)],axis=1)
            meal_df = pd.concat([d_et, d_ms], axis=0).reset_index(drop=True)
            
            meal_df2 = pd.DataFrame(columns=["start_seconds", "end_seconds", "utensil-code"])
            for index, row in meal_df.iterrows():
                smstart = (parse(row["start"])-baseline_date).total_seconds()
                smend = (parse(row["end"])-baseline_date).total_seconds()
                utsl = str(row["utensil"]).split(" ")[-1]
                if utsl == 'nan':
                    utsl = 0
                x = {
                    "start_seconds":smstart,
                    "end_seconds":smend,
                    "utensil-code":utsl
                }
                meal_df2 = meal_df2.append(x, ignore_index=True)
            
            meals[d] = pd.concat([meal_df, meal_df2], axis=1)
        return meals
        
        
    def getMotion(self):
        baseline_date = self.cfg['reference_time']
        sample_rate = self.cfg['sample_rate']
        mt_dfs = {}
        columns=['row','accl_x','accl_y','accl_z','gyro_x','gyro_y','gyro_z']
        
        meals = self.getMeals()
        
        for d in self.dates:
            d_mt_df = pd.DataFrame(columns=columns+['ts'])
            for session in self.mt[self.mt["date"]==d]["path"]:
                userid, datatype, timestamp = session.split('.')[0].split('_')
                motion_df = pd.read_csv(session, header=None, names=columns, sep=",", engine='python')
                motion_df['ts'] = (motion_df["row"]-1)/sample_rate + (parse(timestamp) - baseline_date).total_seconds()
                d_mt_df = d_mt_df.append(motion_df, ignore_index=False).reset_index(drop=True)
            
            meal_df = meals[d]
            d_mt_df["tag"] = 0
            d_mt_df["utsl"] = np.nan
            
            for index, row in meal_df.iterrows():
                mstart = (parse(row["start"])-baseline_date).total_seconds()
                mend = (parse(row["end"])-baseline_date).total_seconds()
                utsl = str(row["utensil"]).split(" ")[-1]
                if utsl == 'nan':
                    utsl = 0
                x = {
                    "start_seconds":mstart,
                    "end_seconds":mend,
                    "utensil":utsl
                }
                d_mt_df.loc[(d_mt_df['ts']>=mstart) & (d_mt_df['ts']<=mend), "tag"] = 1
                d_mt_df.loc[(d_mt_df['ts']>=mstart) & (d_mt_df['ts']<=mend), "utsl"] = int(x["utensil"])
    
            mt_dfs[d] = d_mt_df
        return mt_dfs

    
    def toPandas(self):
        output_df = pd.DataFrame()
        columns=['row','accl_x','accl_y','accl_z','gyro_x','gyro_y','gyro_z']
        
        motion = self.getMotion()

        for d in self.dates:
            d_mt_df = motion[d]
            output_df = output_df.append(d_mt_df)
           
        return output_df.reset_index(drop=True)
    
    
    def draw_day_gram(self, feature):
        baseline_date = self.cfg['reference_time']
        sample_rate = self.cfg['sample_rate']
        
        meals = self.getMeals()
        motion = self.getMotion()
        ticktime = 30
        time_list = [(datetime.strptime("00:00","%H:%M") + timedelta(minutes=m)).strftime("%H:%M") for m in range(0,1440, ticktime)]

        seconds_range = np.asarray(range(60*60*24*sample_rate+1)) / sample_rate
        for d in self.dates:
            d_mt_df = motion[d]
            meal_df = meals[d]
            
            fig = plt.figure(figsize=(30,3))
            x_start = (timezone(self.tzone).localize(datetime.strptime(str(d),"%Y-%m-%d"))-baseline_date).total_seconds()
            full_time_axis = x_start + seconds_range

            d_df = pd.DataFrame(d_mt_df.set_index('ts'), index=full_time_axis)
            plt.plot(d_df.index, d_df[feature])
            x_ticks = d_df.index[::ticktime*60*sample_rate]
            plt.title(str(d).split("T")[0])
            plt.xticks(x_ticks, time_list)
            plt.xlim(x_start, x_start+3600*24)

            for index, row in meal_df.iterrows():
                plt.axvspan(row["start_seconds"],row["end_seconds"],color="red",alpha=0.5)
            
            
            
        
