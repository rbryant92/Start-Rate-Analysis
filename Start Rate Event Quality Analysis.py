#!/usr/bin/env python
# coding: utf-8

# ### Motivation

# An initiative in Q3 2022 was to investigate start rate performance for CTV, OTT, and video lineitems.   Start rate is defined as number of start events for a video ad divided by number of impressions served.  This was to address two primary concerns.  The first concern was over certain apps having unexpectedly poor start rate performance.   This contributes to the second concern, as start rate performance needs to be strong in order to run cost per completed view (CPCV) campaigns.   Over the course of this investigative analysis, we will utilize both internal and third party data.   Some data prior to the end of Q1 2022 have a bug where certain lineitems register start rates greater than 1.0, which is impossible by definition.   Therefore all analysis must investigate data from Q2 2022 or later.
# 
# Things to look at:
# 
# - Start rate of campaign_stats_rollup vs. raw impression
# - Look at evtEventQualityCodes distribution
# - Are these different for different serving groups/device types?
# 
# We will look at event quality codes and create a logic where we eliminate invalid codes manually.  We will confirm this start rate matches (more closely) the start rates seen in CSR.   

# ### Bottom line up front

# - Our top concern when we began to investigate start rate was why some of them were so low.
# - Previously, we found significant discrepancy between the campaign stats rollup (CSR) and the 3rd party rollup.   This was cause for concern.   CSR showed much lower start rates than the 3rd party rollup.
# - We later turned to raw impression data for clues.   A separate discrepancy was found in that notebook: there seemed to be discrepancy between CSR start events and start events in the raw impression data.   Again, we saw much higher start rate in raw impression data than in the CSR, which was due to event quality.
# - The event cleansing logic utilized in this notebook using a breakdown of quality codes in terms of powers of 2 was used to detect invalid start events.  While it is NOT PERFECT (sometimes off by fractions of a percentage point), there is now a much better and very close match between raw and CSR data, and this allows us to use data at the raw impression level to see if any particular impression level features (i.e. 'bidDeviceUserAgent' fields) are contributing to poor start rate.
# - A small number of event codes are causing over 90% of invalid start events.   These seem to be highly driven by the IP_MISMATCH code.   Summaries are presented to see proportions of how often each type of IP mismatch (bidIP, dirIP, and various eventIP's) occur, by browser, device, and exchange.
# - It is not clear how further to investigate third party start rates.  Because BEFORE the event cleansing logic on raw impressions, 3rd party was close to matching the raw impression data which counted invalid events, but then AFTER this event cleansing logic it was close to the CSR where this step has been applied, there is almost certainly a discrepancy in how invalid events are filtered out when third party data is rolled up.   I think we could validate this further if we had 3rd party impression data, but until then we may just be making educated guesses and may need to seek out additional feedback or data sources.

# In[ ]:


import platform_toolkit as ptk
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly_express as px

pd.set_option('display.max_columns',50)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.mode.chained_assignment = None

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ptk.init_auth()
conf = ptk.tshirt('xl')
conf


# In[ ]:


(sc, ss, isp) = ptk.init_platform('start-rate-analysis', 
                                  env_archive=ptk.resolve_conda_env('analysis-preview-py3'))


# In[ ]:


from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import *
from pyspark.sql.functions import explode
from pyspark.sql.functions import lit
from pyspark.sql.window import Window

import platform_toolkit_spark as pts


# In[ ]:


# Isolate the part of Amanda's script for timestamps - to ensure consistency as we load from the various datasets
import minion.services.config as cfg
import mpy.utils as utils
from campaign_utils import minion as mu

li = cfg.LineItemConfigClient()
campaignClient = cfg.CampaignConfigClient()

ts_1 = utils.now_unix_millis() - int(utils.DAY_MS*30)
ts_2 = utils.now_unix_millis() + int(utils.DAY_MS*1)
now_millis = utils.now_unix_millis()


# Pull impression and CSR data:

# In[ ]:


# Dataset produced by https://streamsdash.valassisdigital.net/stream/4121

isp.add_input_full(
    dataset='impression2/campaign_stats_rollup',
    root='REEF_TEAM_AE',
    startTs = ts_1,
    endTs = ts_2,
    key='campaign-stats-rollup-final-stream'
)
campaign_stats = isp.source('campaign-stats-rollup-final-stream')


# In[ ]:


# Dataset produced by https://streamsdash.valassisdigital.net/stream/4150

isp.add_input_full(
    dataset='advertising-logs/impression',
    root='REEF_IMPRESSION_DATA',
    startTs = ts_1,
    endTs = ts_2,
    key='impression-final-stream'
)
impressions = isp.source('impression-final-stream')


# In[ ]:


# Dataset produced by https://streamsdash.valassisdigital.net/stream/36711622 - revisit later

isp.add_input_full(
    dataset='third-party-tagset/third_party_tagset_rollup',
    root='PRODUCT_TP_TAGSET',
    startTs = ts_1,
    endTs = ts_2,
    key='third-party-tagset-rollup-stream'
)
item_group_rollup = isp.source('third-party-tagset-rollup-stream')


# We need to pull from config service in order to join to the third party dataset.   The ts_1 and ts_2 timestamp filters may not work otherwise.

# In[ ]:


with li: 
    
    # facets to include 
    LineItemConstants = li.Constants
    cfg_types = cfg.load_service('Types')
    
    facets = (LineItemConstants.LINE_ITEM_BASICS_FACET |
          LineItemConstants.LINE_ITEM_CAMPAIGN_FACET |
          LineItemConstants.LINE_ITEM_SERVING_FACET |
          LineItemConstants.LINE_ITEM_SERVING_GROUPS_FACET |
          LineItemConstants.LINE_ITEM_SCORED_MAPS_FACET |
          LineItemConstants.LINE_ITEM_CLIENT_IO_FACET |
          LineItemConstants.LINE_ITEM_CLIENT_BENCHMARKS_FACET
         )
    
    start_time_filter = cfg.minion_types.QueryFilter()
    start_time_filter.field = li.Constants.FILTER_FIELD_START_TIME
    start_time_filter.operation = cfg.minion_types.FILTER_OPERATION_GREATER_THAN_EQUALS
    start_time_filter.values = [cfg.minion_types.ThriftValue(dateValue=ts_1)]
    
    end_time_filter = cfg.minion_types.QueryFilter()
    end_time_filter.field = li.Constants.FILTER_FIELD_END_TIME
    end_time_filter.operation = cfg.minion_types.FILTER_OPERATION_LESS_THAN_EQUALS
    end_time_filter.values = [cfg.minion_types.ThriftValue(dateValue=ts_2)]
    
    state_filter = cfg.minion_types.QueryFilter()
    state_filter.field = li.Constants.FILTER_FIELD_STATE
    state_filter.operation = cfg.minion_types.FILTER_OPERATION_EQUALS
    state_filter.values = [cfg.minion_types.ThriftValue(intValue=campaignClient.Types.ItemState.COMPLETED)]
    
    region_filter = cfg_types.QueryFilter(cfg.LineItemConfigClient.Constants.FILTER_FIELD_REGION_ID,
                                      cfg.minion_types.FILTER_OPERATION_EQUALS,
                                      [cfg.minion_types.ThriftValue(intValue=0)])
    
    filters = [start_time_filter, end_time_filter, state_filter, region_filter]
        
    line_item_config_list = []
    line_chunk = li.fetchLineItemsByFiltersInChunks(facets, filters, previousItemId=0,
                                                    chunkSize=1000, agentParamsToInclude={}).lineItems
    line_item_config_list.extend(line_chunk)
    
    if len(line_chunk) > 1:
        while True:
            line_chunk = li.fetchLineItemsByFiltersInChunks(facets, filters, 
                                                            previousItemId=line_chunk[len(line_chunk) - 1].itemId,
                                                            chunkSize=1000, agentParamsToInclude={}).lineItems
            line_item_config_list.extend(line_chunk)
            if len(line_chunk) < 1000:
                break


# In[ ]:


# Build dictionary with success metric (benchmark) names
with cfg.ClientBenchmarkConfigClient() as cbcc:
        benchmarks = cbcc.fetchSuccessMetricAttributes()
success_metrics = {c:v.metricName for (c,v) in benchmarks.items()}

def millis_to_dt(ts_millis):
    """
    Convert millis into datetime
    """
    return dt.datetime.fromtimestamp(ts_millis/1000.0)
    
def read_parameter(k, params):
    if params.paramSetValues and k in params.paramSetValues:
        return params.paramSetValues[k].paramValue
    else:
        return None

def flight_length(start, end):
    """
    Return length of flight in days, minimum of 1 day
    """
    start_dt = millis_to_dt(start)
    end_dt = millis_to_dt(end)
    delta = end_dt - start_dt
    return np.maximum(1,delta.days)
    
def extract_line_item_fields(li_config):
    """
    Pull and process lineitem fields from config service
    """
    line_item_id = li_config.itemId
    state = li_config.itemBasics.state
    media_type = cfg.LineItemConfigClient.Types.AdClass._VALUES_TO_NAMES[li_config.itemBasics.adClass]
    product_id = li_config.itemClientIO.productId
    start_time = li_config.itemBasics.startTime
    end_time = li_config.itemBasics.endTime
    flight_days = flight_length(start_time, end_time)
    imp_goal = li_config.itemServing.impressions
    placement_types = li_config.itemServing.placementType
    is_mobile = li_config.itemServing.mobile
    device_type = li_config.itemServing.deviceTypes
    is_locationBased = li_config.itemServing.locationBased
    
    return (line_item_id, state, media_type, product_id, start_time, end_time, flight_days, imp_goal, 
            placement_types, is_mobile, device_type, is_locationBased)


# In[ ]:


line_item_config_extracted = [extract_line_item_fields(li) for li in line_item_config_list]

li_config = pd.DataFrame(line_item_config_extracted, columns=['lineItemId', 'state', 'adClass', 'productId', 
                                                              'startTime', 'endTime', 'flightDays','impressionGoal', 
                                                              'placementType', 'isMobile', 'deviceType', 
                                                              'isLocationBased'])

lineItems = [x.itemId for x in line_item_config_list]


# In[ ]:


# Manipulate the CSR to summarize impressions and start events
campaign_stats_rollup = (campaign_stats
                        .filter(F.col("lineItemId").isin(lineItems))
                        .filter(F.col('adWidth')==0)
                        .filter(F.col('adHeight')==0)
                        .withColumn('videoStart', F.col('eventCounts').getItem('start'))
                        .groupby('lineItemId')
                        .agg(F.sum('numImpressions').alias('sumImpressionsInternal'),
                             F.sum('videoStart').alias('startEventsInternal'))
                        )


# In[ ]:


# Manipulate 3rd party data to summarize impressions and start events
item_group_rollup = (item_group_rollup
                    .filter(F.col('size')=='0x0')
                    .filter(F.col('vendorType')=='tag')
                    .select("timevalueFrom",
                            "timevalueTo",
                            "eligibleItems", 
                            "numImpressions", 
                            F.explode("miscFields").alias("eventKey", "eventCountStr"))
                    .filter(F.lower(F.col("eventKey")).isin(["video plays"]))
                    .withColumn("eventCount", F.col("eventCountStr").cast(T.IntegerType()))
                    .filter(F.size("eligibleItems") == 1)
                    .withColumn("lineItemId", F.explode("eligibleItems"))
                    .filter(F.col("lineItemId").isin(lineItems))
                    .groupby('lineItemId')
                    .agg(F.sum('numImpressions').alias('sumImpressionsThirdParty'),
                         F.sum('eventCount').alias('startEventsThirdParty'))
                    )


# In[ ]:


from pyspark.sql.types import (ArrayType, StructType, StringType, StructField)

# UDF for use in exploding the "eventType" and "eventQuality" fields at the same time
combine = F.udf(lambda a, b, c, d, e: list(zip(a, b, c, d, e)),
              ArrayType(StructType([StructField("eventType", StringType()),
                                    StructField("eventQuality", StringType()),
                                    StructField("evtIpFromRequest", StringType()),
                                    StructField("evtIpFromUrl", StringType()),
                                    StructField("evtRawIp", StringType())])))

# SDF for instances where there are start events
startEventDF = (impressions
                .filter(F.col('dirLineItemId').isin(lineItems))
                .filter(F.col('impTrackingClass') > 0)
                .withColumn('eventType', F.col('impEvents').getItem('evtEventType'))
                .withColumn('eventQuality', F.col('impEvents').getItem('evtQuality'))
                .withColumn('evtIpFromRequest', F.col('impEvents').getItem('evtIpFromRequest'))
                .withColumn('evtIpFromUrl', F.col('impEvents').getItem('evtIpFromUrl'))
                .withColumn('evtRawIp', F.col('impEvents').getItem('evtRawIp'))
                .withColumn('new', combine('eventType', 'eventQuality', 'evtIpFromRequest', 'evtIpFromUrl', 'evtRawIp'))
                .withColumn("new", F.explode("new"))
                .select('dirLineItemId',
                        'impServingGroupId',
                        'bidIp',
                        'dirIp',
                        F.col('new.eventType').alias('eventType'),
                        F.col('new.eventQuality').alias('eventQuality'),
                        F.col('new.evtIpFromRequest').alias('evtIpFromRequest'),
                        F.col('new.evtIpFromUrl').alias('evtIpFromUrl'),
                        F.col('new.evtRawIp').alias('evtRawIp'),
                        'bidUserAgentLookup')
                .withColumnRenamed("dirLineItemId", "lineItemId")
                .filter(F.col('eventType') == 'start')
                .withColumn("bidRtbDeviceType", F.col("bidUserAgentLookup").getItem("bidRtbDeviceType"))
                .withColumn("bidUserAgentScale", F.col("bidUserAgentLookup").getItem("bidUserAgentScale"))
                .withColumn("Flags", F.col("bidUserAgentLookup").getItem("Flags"))
                .withColumn("bidBrowserName", F.col("bidUserAgentLookup").getItem("bidBrowserName"))
                .withColumn("bidDeviceOS", F.col("bidUserAgentLookup").getItem("bidDeviceOS"))
                .withColumn("bidDeviceOSVersion", F.col("bidUserAgentLookup").getItem("bidDeviceOSVersion"))
                .withColumn("bidDeviceMake", F.col("bidUserAgentLookup").getItem("bidDeviceMake"))
                .drop("bidUserAgentLookup")
                .withColumn("eventQuality", F.col("eventQuality").cast(T.IntegerType()))
               )

# Need to convert this SDF to counts grouped by "eventQuality"
startEventGrp = (startEventDF
                .groupby('eventQuality')
                .agg(F.count("*").alias("eventTypeCount"))
                )


# In[ ]:


# Need a separate SDF indicating impressions where there is NO start event
# We will create a "startEvent" field and union this to the positive start event DF later
nonStartDF = (impressions
              .filter(F.col('dirLineItemId').isin(lineItems))
              .filter(F.col('impTrackingClass') > 0)
              .withColumn('eventType', F.col('impEvents').getItem('evtEventType'))
              .withColumn('eventQuality', F.col('impEvents').getItem('evtQuality'))
              .withColumn('evtIpFromRequest', F.col('impEvents').getItem('evtIpFromRequest'))
              .withColumn('evtIpFromUrl', F.col('impEvents').getItem('evtIpFromUrl'))
              .withColumn('evtRawIp', F.col('impEvents').getItem('evtRawIp'))
              .select("dirLineItemId",
                      "impServingGroupId",
                      "bidIp",
                      "dirIp",
                      "evtIpFromRequest",
                      "evtIpFromUrl",
                      "evtRawIp",
                      "eventType",
                      "eventQuality",
                      "bidUserAgentLookup")
              .withColumnRenamed("dirLineItemId", "lineItemId")
              .filter(~array_contains(col("eventType"),"start"))
              .withColumn("bidRtbDeviceType", F.col("bidUserAgentLookup").getItem("bidRtbDeviceType"))
              .withColumn("bidUserAgentScale", F.col("bidUserAgentLookup").getItem("bidUserAgentScale"))
              .withColumn("Flags", F.col("bidUserAgentLookup").getItem("Flags"))
              .withColumn("bidBrowserName", F.col("bidUserAgentLookup").getItem("bidBrowserName"))
              .withColumn("bidDeviceOS", F.col("bidUserAgentLookup").getItem("bidDeviceOS"))
              .withColumn("bidDeviceOSVersion", F.col("bidUserAgentLookup").getItem("bidDeviceOSVersion"))
              .withColumn("bidDeviceMake", F.col("bidUserAgentLookup").getItem("bidDeviceMake"))
              .drop("bidUserAgentLookup", "eventQuality")
              .withColumn("startEvent", lit("None"))
              )


# In[ ]:


# Convert the grouped positive data frame to pandas to create invalid event code logic
startEventPd = startEventGrp.toPandas()


# In[ ]:


# Create a function to break each code down in terms of the powers of 2 (including 2**0 = 1)
invalidCodes = (0, 4, 128, 256, 512, 8192, 32768, 2097152)

def bitBreak(x):
    powers = []
    i = 1
    while i <= x:
        if i & x:
            powers.append(i)
        i <<= 1
    return powers

# If an invalid code is included in the total breakdown for the quality code, the event is INVALID
def validDefinition(x):
    if any(c in invalidCodes for c in x):
        result = "Invalid"
    else:
        result = "Valid"
    return result

# New function for ip mismatch specifically
def ipMismatch(x):
    if any(c == 512 for c in x):
        result = True
    else:
        result = False
    return result

startEventPd["eventBreakdown"] = startEventPd["eventQuality"].apply(lambda x: bitBreak(x))
startEventPd["startEvent"] = startEventPd["eventBreakdown"].apply(lambda x: validDefinition(x))
startEventPd["ipMismatch"] = startEventPd["eventBreakdown"].apply(lambda x: ipMismatch(x))


# In[ ]:


# Look at the head of this DF
startEventPd.sort_values(by = "eventTypeCount", ascending = False).head(30)


# In[ ]:


# What percentage of the time is this occurring??
invalidEvents = startEventPd[startEventPd.startEvent == "Invalid"]
ipMismatches = startEventPd[startEventPd.ipMismatch == True]

total_invalid = invalidEvents.eventTypeCount.sum()
total_mismatch = ipMismatches.eventTypeCount.sum()

# Ratio
total_mismatch / total_invalid


# Instances with the code 512 (IP_MISMATCH) are causing a tremendous number of start events to be invalid.  Many of these also have code 2 (BROWSER_ID_WEAK_MATCH) or code 64 (MID2_MISSING).   These codes appear to make up about >90% of invalid events; therefore, these are the low hanging fruit for which it may be worth seeing if they appear disproportionately for certain devices/browsers, etc.

# In[ ]:


# Convert this pandas df back to Spark with isolated necessary columns
eventQualityLookup = ss.createDataFrame(startEventPd[["eventQuality", "eventBreakdown", "startEvent"]])

# Join this lookup table back to the original df - we will drop "event quality" and "event breakdown" after we look
# at IP mismatches
startEventDFwithLookup = (startEventDF
                         .join(eventQualityLookup, on = "eventQuality")
                         #.drop("eventQuality", "eventBreakdown")
                         )


# In[ ]:


# We will now have a full reconstructed impression dataset - categorized with startEvent = None, Invalid, or Valid
# We can get rid of the eventType and other fields to do this
startEventDFwithLookup = startEventDFwithLookup.drop("eventType", "eventQuality", "eventBreakdown")
nonStartDF = nonStartDF.drop("eventType")

# Because we don't care about events that aren't starts, let's turn these variables into null strings
nonStartDF = (nonStartDF
             .withColumn('evtIpFromRequest', lit(None).cast(StringType()))
             .withColumn('evtIpFromUrl', lit(None).cast(StringType()))
             .withColumn('evtRawIp', lit(None).cast(StringType()))
             )

# Union all instead of union these datasets to bring them together and ensure identical rows are retained
fullStartDF = (startEventDFwithLookup.unionAll(nonStartDF))


# In[ ]:


fullStartDF = (fullStartDF
              .withColumn("invalidStart", F.when(fullStartDF.startEvent == "Invalid", 1).otherwise(0))
              .withColumn("noStart", F.when(fullStartDF.startEvent == "None", 1).otherwise(0))
              .withColumn("validStart", F.when(fullStartDF.startEvent == "Valid", 1).otherwise(0))
              )


# In[ ]:


fullStartDF = (fullStartDF
              .withColumn("bidLiMismatch", F.when(F.col("bidIp") != F.col("dirIp"), 1).otherwise(0))
              .withColumn("liUrlMismatch", F.when(F.col("dirIp") != F.col("evtIpFromUrl"), 1).otherwise(0))
              .withColumn("liRawMismatch", F.when(F.col("dirIp") != F.col("evtRawIp"), 1).otherwise(0))
              )


# In[ ]:


fullStartDF.show(10)


# In[ ]:


validBreakdown = (fullStartDF
              .filter(F.col("validStart") == 1)
              .agg(F.count("*").alias("totalCount"),
                   F.sum("bidLiMismatch").alias("bidLiMismatches"),
                   F.sum("liUrlMismatch").alias("liUrlMismatches"),
                   F.sum("liRawMismatch").alias("liRawMismatches")
                  )
               .withColumn("bidLiProp", F.col("bidLiMismatches") / F.col("totalCount"))
               .withColumn("liUrlProp", F.col("liUrlMismatches") / F.col("totalCount"))
               .withColumn("liRawProp", F.col("liRawMismatches") / F.col("totalCount"))
               .drop("bidLiMismatches", "bidRequestMismatches", "bidUrlMismatches", "bidRawMismatches",
                    "liRequestMismatches", "liUrlMismatches", "liRawMismatches")
              )
validBreakdown.show(20)


# In[ ]:


invalidBreakdown = (fullStartDF
              .filter(F.col("invalidStart") == 1)
              .agg(F.count("*").alias("totalCount"),
                   F.sum("bidLiMismatch").alias("bidLiMismatches"),
                   F.sum("liUrlMismatch").alias("liUrlMismatches"),
                   F.sum("liRawMismatch").alias("liRawMismatches")
                  )
               .withColumn("bidLiProp", F.col("bidLiMismatches") / F.col("totalCount"))
               .withColumn("liUrlProp", F.col("liUrlMismatches") / F.col("totalCount"))
               .withColumn("liRawProp", F.col("liRawMismatches") / F.col("totalCount"))
               .drop("bidLiMismatches", "bidRequestMismatches", "bidUrlMismatches", "bidRawMismatches",
                    "liRequestMismatches", "liUrlMismatches", "liRawMismatches")
              )
invalidBreakdown.show(20)


# ### Brief investigation of the IP mismatches

# To reiterate, instances with IP mismatches (code 512) constitute a majority of invalid events which drive start rate performance lower.  We will look at the frequency of this occurrence by various other bid level features.

# In[ ]:


deviceTypes = (fullStartDF
              .groupBy("bidRtbDeviceType")
              .agg(F.count("*").alias("totalCount"),
                   F.sum("bidLiMismatch").alias("bidLiMismatches"),
                   F.sum("liUrlMismatch").alias("liUrlMismatches"),
                   F.sum("liRawMismatch").alias("liRawMismatches")
                  )
               .withColumn("bidLiProp", F.col("bidLiMismatches") / F.col("totalCount"))
               .withColumn("liUrlProp", F.col("liUrlMismatches") / F.col("totalCount"))
               .withColumn("liRawProp", F.col("liRawMismatches") / F.col("totalCount"))
               .drop("bidLiMismatches", "bidRequestMismatches", "bidUrlMismatches", "bidRawMismatches",
                    "liRequestMismatches", "liUrlMismatches", "liRawMismatches")
              )
deviceTypes = pts.to_pandas_parallel(deviceTypes)
deviceTypes.sort_values(by = "totalCount", ascending = False).head(20)


# In[ ]:


browserTypes = (fullStartDF
              .groupBy("bidDeviceOS")
              .agg(F.count("*").alias("totalCount"),
                   F.sum("bidLiMismatch").alias("bidLiMismatches"),
                   F.sum("liUrlMismatch").alias("liUrlMismatches"),
                   F.sum("liRawMismatch").alias("liRawMismatches")
                  )
               .withColumn("bidLiProp", F.col("bidLiMismatches") / F.col("totalCount"))
               .withColumn("liUrlProp", F.col("liUrlMismatches") / F.col("totalCount"))
               .withColumn("liRawProp", F.col("liRawMismatches") / F.col("totalCount"))
               .drop("bidLiMismatches", "bidRequestMismatches", "bidUrlMismatches", "bidRawMismatches",
                    "liRequestMismatches", "liUrlMismatches", "liRawMismatches")
              )
browserTypes = pts.to_pandas_parallel(browserTypes)
browserTypes.sort_values(by = "totalCount", ascending = False).head(20)


# In[ ]:


deviceMakes = (fullStartDF
              .groupBy("bidDeviceMake")
              .agg(F.count("*").alias("totalCount"),
                   F.sum("bidLiMismatch").alias("bidLiMismatches"),
                   F.sum("liUrlMismatch").alias("liUrlMismatches"),
                   F.sum("liRawMismatch").alias("liRawMismatches")
                  )
               .withColumn("bidLiProp", F.col("bidLiMismatches") / F.col("totalCount"))
               .withColumn("liUrlProp", F.col("liUrlMismatches") / F.col("totalCount"))
               .withColumn("liRawProp", F.col("liRawMismatches") / F.col("totalCount"))
               .drop("bidLiMismatches", "bidRequestMismatches", "bidUrlMismatches", "bidRawMismatches",
                    "liRequestMismatches", "liUrlMismatches", "liRawMismatches")
              )
deviceMakes = pts.to_pandas_parallel(deviceMakes)
deviceMakes.sort_values(by = "totalCount", ascending = False).head(20)


# In[ ]:


servingGroups = (fullStartDF
                .groupBy("impServingGroupId")
                .agg(F.count("*").alias("totalCount"),
                     F.sum("bidLiMismatch").alias("bidLiMismatches"),
                     F.sum("liUrlMismatch").alias("liUrlMismatches"),
                     F.sum("liRawMismatch").alias("liRawMismatches")
                    )
                 .withColumn("bidLiProp", F.col("bidLiMismatches") / F.col("totalCount"))
                 .withColumn("liUrlProp", F.col("liUrlMismatches") / F.col("totalCount"))
                 .withColumn("liRawProp", F.col("liRawMismatches") / F.col("totalCount"))
                 .drop("bidLiMismatches", "bidRequestMismatches", "bidUrlMismatches", "bidRawMismatches",
                      "liRequestMismatches", "liUrlMismatches", "liRawMismatches")
                )
servingGroups = pts.to_pandas_parallel(servingGroups)


# In[ ]:


# Map exchange names
serving_dict = mu.get_serving_group_dict()

servingGroups['adExchangeName'] = servingGroups['impServingGroupId'].map(serving_dict)
servingGroups.sort_values(by = "totalCount", ascending = False).head(20)


# #### Join the datasets

# In[ ]:


imps_line_level = (fullStartDF
                  .groupby('lineItemId')
                  .agg(F.count("*").alias("countImpressionEvents"),
                       F.sum("validStart").alias("countStartEvents"))
                  )


# In[ ]:


startRateLI = campaign_stats_rollup.join(imps_line_level, on = "lineItemId")
startRateLI = startRateLI.join(item_group_rollup, on = "lineItemId")


# In[ ]:


startRateLI.show(5)


# In[ ]:


startRateLI = pts.to_pandas_parallel(startRateLI)


# In[ ]:


# Create new columns
startRateLI["startRateInternal"] = startRateLI["startEventsInternal"] / startRateLI["sumImpressionsInternal"]
startRateLI["rawStartRate"] = startRateLI["countStartEvents"] / startRateLI["countImpressionEvents"]
startRateLI["startRateThirdParty"] = startRateLI["startEventsThirdParty"] / startRateLI["sumImpressionsThirdParty"]


# In[ ]:


# Let's get rid of lineitems with <1000 impressions for this... these are probably test items
startRateLI = startRateLI[startRateLI.sumImpressionsInternal > 1000]


# In[ ]:


startRateLI.describe()


# In[ ]:


startRateLI.info()


# ### Start rate discrepancy - investigation

# This looks at our data in its totality, so that we can compare the three data sources: raw, CSR, and third party.

# In[ ]:


fig = px.scatter(startRateLI, x="sumImpressionsInternal", y="countImpressionEvents", log_x = False,
                 log_y = False, hover_data=['lineItemId'], title = "Internal vs. Raw Impressions")
fig.show()


# In[ ]:


fig = px.scatter(startRateLI, x="startEventsInternal", y="countStartEvents", log_x = False,
                 log_y = False, hover_data=['lineItemId'], title = "Internal vs. Raw Impression Data Start Events")
fig.show()


# In[ ]:


fig = px.scatter(startRateLI, x="startRateInternal", y="rawStartRate", log_x = False,
                 log_y = False, hover_data=['lineItemId'], title = "Internal vs. Raw Impression Start Rate")
fig.show()


# In[ ]:


fig = px.scatter(startRateLI, x="sumImpressionsThirdParty", y="countImpressionEvents", log_x = False,
                 log_y = False, hover_data=['lineItemId'], 
                 title = "3rd Party vs. Raw Impressions")
fig.show()


# In[ ]:


fig = px.scatter(startRateLI, x="countStartEvents", y="startEventsThirdParty", log_x = True,
                 log_y = True, hover_data=['lineItemId'], title = "Internal vs. 3rd Party Start Events")
fig.show()

