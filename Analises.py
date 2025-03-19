###### Case Deep Dive e Propensão a churn para uma locadora de veiculos ficticia ######

# import e dataframe
import pandas as pd
df = pd.read_csv("C:\\Users\\Userv\\OneDrive\\Documentos\\Kovi\\query.csv")

# Declarando variáveis
driver_id = df['driver_id']
date = df['date']
kind = df['kind']
amount = df['amount']

# 1) Transformando o dataframe
# Converter a coluna 'date' para datetime
df['date'] = pd.to_datetime(df['date'])

# Calcular AgeWeek
df_age = df.groupby('driver_id')['date'].agg(['min', 'max']).reset_index()
df_age['AgeWeek'] = ((df_age['max'] - df_age['min']).dt.days / 7).round(2)

# Calcular TotalAmount
df_amount = df.groupby('driver_id')['amount'].sum().reset_index()
df_amount = df_amount.rename(columns={'amount': 'TotalAmount'})

# Calcular Status
df_status = df.groupby('driver_id')['date'].max().reset_index()
df_status['Status'] = df_status['date'].apply(
    lambda x: 'Ativo' if x >= pd.to_datetime('2024-12-01') else 'Inativo')

# Calcular Kind_upgrade
df_kind_upgrade = df.groupby('driver_id')['kind'].apply(
    lambda x: 1 if 'FIRST_PAYMENT_EXCHANGE' in x.values else 0).reset_index()
df_kind_upgrade = df_kind_upgrade.rename(columns={'kind': 'Kind_upgrade'})

# Calcular datachurn
inativos_df = df_status[df_status['Status'] == 'Inativo'].copy()
datachurn_df = df[df['driver_id'].isin(inativos_df['driver_id'])].groupby('driver_id')[
    'date'].max().reset_index()
datachurn_df = datachurn_df.rename(columns={'date': 'datachurn'})

# Calcular UpgradeDate
upgrade_date_df = df[df['kind'] == 'FIRST_PAYMENT_EXCHANGE'].groupby('driver_id')[
    'date'].min().reset_index()
upgrade_date_df = upgrade_date_df.rename(columns={'date': 'UpgradeDate'})

# Calcular UpgradeWeeks
first_payment_df = df[df['kind'] == 'FIRST_PAYMENT'].groupby('driver_id')[
    'date'].min().reset_index()
first_payment_df = first_payment_df.rename(columns={'date': 'FirstPaymentDate'})
upgrade_weeks_df = pd.merge(first_payment_df, upgrade_date_df, on='driver_id', how='inner')
upgrade_weeks_df['UpgradeWeeks'] = (
        (upgrade_weeks_df['UpgradeDate'] - upgrade_weeks_df['FirstPaymentDate']).dt.days / 7).round(
    2)
upgrade_weeks_df['UpgradeWeeks'] = upgrade_weeks_df['UpgradeWeeks'].apply(
    lambda x: 0 if x < 0 else x)  # Correção: Garantir que UpgradeWeeks >= 0
upgrade_weeks_df = upgrade_weeks_df[['driver_id', 'UpgradeWeeks']]

# Criar o DataFrame final
drivers_df = pd.merge(df_age, df_amount, on='driver_id', how='outer')
drivers_df = pd.merge(drivers_df, df_status, on='driver_id', how='outer')
drivers_df = pd.merge(drivers_df, df_kind_upgrade, on='driver_id', how='outer')
drivers_df = pd.merge(drivers_df, datachurn_df, on='driver_id', how='left')
drivers_df = pd.merge(drivers_df, upgrade_date_df, on='driver_id', how='left')
drivers_df = pd.merge(drivers_df, upgrade_weeks_df, on='driver_id', how='left')

# Calcular o LTV como TotalAmount / AgeWeek
drivers_df['LTV'] = drivers_df['TotalAmount'] / drivers_df['AgeWeek']

# Remover valores infinitos ou nulos do LTV
drivers_df = drivers_df.replace([float('inf'), -float('inf')], 0)
drivers_df['LTV'] = drivers_df['LTV'].fillna(0)

# Classificar os drivers em quartis com base no LTV
drivers_df['quartil'] = pd.qcut(drivers_df['LTV'], 4, labels=[1, 2, 3, 4])

# Calcular NroChanges
nro_changes_df = df[df['kind'] == 'FIRST_PAYMENT_EXCHANGE'].groupby('driver_id').size().reset_index(name='NroChanges')
drivers_df = pd.merge(drivers_df, nro_changes_df, on='driver_id', how='left')
drivers_df['NroChanges'] = drivers_df['NroChanges'].fillna(0)

# Selecionar as colunas desejadas
drivers_df = drivers_df[
    ['driver_id', 'AgeWeek', 'TotalAmount', 'Status', 'Kind_upgrade', 'datachurn',
     'UpgradeDate', 'UpgradeWeeks', 'LTV', 'quartil', 'NroChanges']]

print(drivers_df.head())

# Exportar DataFrame para excel para fazer os gráficos
drivers_df.to_csv("driver_data7.csv", index=False)

# 2) Modelo de Regressão Logística

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Etapa 2: Regressão Logística
drivers_df['Status'] = drivers_df['Status'].apply(lambda x: 1 if x == 'Ativo' else 0)

# Verifique os nomes das colunas para garantir que 'Kind_upgrade' e 'NroChanges' existem
print(drivers_df.columns)

# Certifique-se de que os nomes das colunas estão corretos (sensíveis a maiúsculas e minúsculas)
X = drivers_df[['TotalAmount', 'AgeWeek', 'Kind_upgrade', 'LTV', 'NroChanges']]
y = drivers_df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Relatório de Classificação:\n", report)

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
report_df.to_csv("metrics_churn_score.csv", index=True)
print("Métricas do modelo exportadas para metrics_churn_score.csv")

score_df = pd.DataFrame({'driver_id': drivers_df.loc[y_test.index, 'driver_id'], 'churn_probability': y_prob})
score_df['churn_probability'] = (score_df['churn_probability'] * 1000).round().astype(int)
score_df = score_df.sort_values(by='churn_probability', ascending=False)

# Etapa 3: Junção e Exportação
drivers_df['driver_id'] = drivers_df['driver_id'].astype(str)
score_df['driver_id'] = score_df['driver_id'].astype(str)

drivers_df = pd.merge(drivers_df, score_df, on='driver_id', how='left')

drivers_df.to_csv("drivers_analysis_with_churn_score.csv", index=False)
print("Resultados finais exportados para drivers_analysis_with_churn_score.csv")

# Calcular as métricas do modelo
report = classification_report(y_test, y_pred)

# Exibir as métricas do modelo
print(report)

# Exportar as métricas do modelo
with open("report_churn_score.csv", "w") as file:
    file.write(report)

print("Relatório de classificação salvo em report_churn_score.csv")

# 3) Fatiou... passou.

# Etapa 3: Junção e Exportação
drivers_df['driver_id'] = drivers_df['driver_id'].astype(str)
score_df['driver_id'] = score_df['driver_id'].astype(str)

# Correção: Resetar o índice de score_df antes de usar loc
score_df = score_df.reset_index(drop=True)

# Correção: Usar o índice de y_test diretamente para acessar score_df
score_df = pd.DataFrame({'driver_id': drivers_df.loc[y_test.index, 'driver_id'].values, 'churn_probability': y_prob})
score_df['churn_probability'] = (score_df['churn_probability'] * 1000).round().astype(int)
score_df = score_df.sort_values(by='churn_probability', ascending=False)


drivers_df['driver_id'] = drivers_df['driver_id'].astype(str)
score_df['driver_id'] = score_df['driver_id'].astype(str)

drivers_df = pd.merge(drivers_df, score_df, on='driver_id', how='left')

drivers_df.to_csv("drivers_analysis_com_score.csv", index=False)
print("drivers_analysis_com_score.csv")
