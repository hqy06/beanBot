"""
Calculate scores for services based on the review dataset.
"""

# ===============================================
# Import Libraries
# ===============================================
import numpy as np
import pandas as pd


# ===============================================
# Global Variables
# ===============================================
DATASET = "dummy.csv"
N_FEATURE = 4
N_SERVICE = 10


# ===============================================
# Metrics and so on
# ===============================================

def _calulate_numercial(scores, type="avg"):
    if type == "avg":
        return np.average(scores)
    elif type == "l2":
        return np.sqrt(np.sum(feature_sscores ** 2) / n_entries)
    else:
        raise ValueError(
            "invalid type, expecting avg or l2, get {}".format(type))


def calculate_scores(features, user_profile, df, weights):
    s_scores, m_scores = [], []
    (s_weight, m_weight) = weights
    for s in range(N_SERVICE):
        service_df = df.loc[df["sID"] == s + 1]
        if len(service_df) == 0:
            print("nothing mathces!")
            s_scores.append("nan")
            m_scores.append("nan")
            continue
        f_sscores, f_mscores = [], []
        # print("into feautres")
        for f in features:
            feature_df = service_df[service_df[f] == user_profile[f]]
            n_entries = len(feature_df)
            if n_entries == 0:
                f_sscores.append(2.5)
                f_mscores.append(2.5)
                continue
            feature_sscores = feature_df["s_score"].values
            feature_mscores = feature_df["m_score"].values
            f_sscores.append(_calulate_numercial(feature_sscores))
            f_mscores.append(_calulate_numercial(feature_mscores))
        s_score = np.dot(f_sscores, s_weight) / N_FEATURE
        m_score = np.dot(f_mscores, m_weight) / N_FEATURE
        # print(s + 1, s_score, m_score)
        s_scores.append(s_score)
        m_scores.append(m_score)
    return s_scores, m_scores


# ===============================================
# Main
# ===============================================


def main():
    df = pd.read_csv(DATASET)
    usr_profile = {"ufeature1": "F", "ufeature2": "fr",
                   "ufeature3": "U3", "ufeaure4": "CA"}

    s_weights = np.random.random_sample(N_FEATURE)
    m_weights = np.random.random_sample(N_FEATURE)

    features = usr_profile.keys()
    s_score, m_score = calculate_scores(
        features, usr_profile, df, (s_weights, m_weights))

    print(s_score, "\n", m_score)


if __name__ == "__main__":
    main()
