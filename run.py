import pandas as pd
from GiniImpurity import GiniImpurityTree, GiniNode  
from WeightedTree import WeightedTree, Node
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':    
    col_names = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result']
    df = pd.read_csv('data.csv', skiprows=1, header=None,names=col_names)
    df['Result'] = df['Result'].replace({-1:0})
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1,1)

    for i in range(35, 46):
        print(f'random_state: {i}')
        start_time = time.time()
        X, a, Y, b = train_test_split(x, y, test_size=0.2,random_state=i)
        X_t, X_test, Y_t, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

        classifier = WeightedTree(X_t, Y_t, min_samples=3, max_depth=12)
        classifier.train(X_t,Y_t)

        classifier.test(X_test, Y_test)

        Y_pred3 = classifier.test2(a, b, 0.5)
        accuracy = accuracy_score(b, Y_pred3)
        print(f'Weighted Tree Accuracy: {accuracy}')
        print(f'Weighted Tree Accurate Predictions: {len(Y_pred3)*accuracy}')
        print(f'Weighted Tree Runtime:{time.time() - start_time}')

        start_time = time.time()
        classifier2 = GiniImpurityTree(min_samples=3, max_depth=12)
        classifier2.train(X_t,Y_t)

        Y_pred4 = classifier2.test(a, b, 0.5)
        accuracy = accuracy_score(b, Y_pred4)
        print(f'Gini Impurity Tree Accuracy: {accuracy}')
        print(f'Gini Impurity Tree Accurate Preditions: {len(Y_pred4)*accuracy}')
        print(f'Gini Impurity Tree Runtime{time.time() - start_time}')
        print(f'Out of {len(b)}\n')