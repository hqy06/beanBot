#!/usr/bin/env python


import pandas as pd

### Input: A csv file containg all services and a csv file storing the profile of the user in the FIRST row
def fixed_feature_filter(services,profile,index=0):
    # Drop null rows
    services=services.dropna(subset=["name"])
    if profile["campus"][index]=="MacDonald":
        services = services[services["location"]!="downtown"]
    if profile["language"][index]=="English":
        services = services[services["language"]!="fr"]
    if profile["language"][index]=="French":
        services = services[services["language"]!="en"]
    if profile["international"][index]=="no":
        services = services[services["name"]!="The Buddy Program"]
    if profile["yearofstudy"][index]!="PhD":
        services = services[services["name"]!="PhD Support Group"]
    return services



services=pd.read_csv("Services.csv")
profile=pd.read_csv("Profiles.csv")

new_services=fixed_feature_filter(services,profile)





