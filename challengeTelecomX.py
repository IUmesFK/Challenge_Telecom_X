import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import pandas as pd
import numpy as np
import requests
import json

url = 'https://raw.githubusercontent.com/alura-cursos/challenge2-data-science-LATAM/refs/heads/main/TelecomX_Data.json'

datos = requests.get('https://raw.githubusercontent.com/alura-cursos/challenge2-data-science-LATAM/refs/heads/main/TelecomX_Data.json')
df_clientes = json.loads(datos.text)
df_clientes_normalizado = pd.json_normalize(df_clientes)

columnas = df_clientes_normalizado.columns

pd.set_option('future.no_silent_downcasting', True)

# customerID = id del cliente, tipo de dato OBJETO
# Churn = Indica si el cliente sigue o no en la empresa, tipo de dato OBJETO, cambiar a BOOL
# customer.gender = sexo del cliente, tipo de dato OBJETO
# customer.SeniorCitizen = indica si el cliente es jubilado o no, tipo de dato INT64, cambiar a BOOL
# customer.Partner = indica si tiene pareja/cónyuge o no, tipo de dato OBJETO, cambiar a BOOL
# customer.Dependents = Indica si tiene personas que dependan economicamente de el (hijos, cónyuge, padres, etc), tipo de dato OBJETO, cambiar a BOOL
# customer.tenure = indica los meses que el cliente ha estado ligado a la empresa
# phone.PhoneService = indica si el cliente tiene o no servicio telefonico, tipo de dato OBJETO, cambiar a BOOL
# phone.MultipleLines = indica si tiene multiples lineas telefonicas o si no tiene servicio telefonico
# internet.InternetService = indica el tipo de servicio de internet que tiene el cliente (DSL o Fibra optica) o si no tiene servicio de internet
# internet.OnlineSecurity = indica si tiene seguridad online o si no tiene servicio de internet
# internet.OnlineBackup = indica si tiene un servicio de copia de seguridad, o si no tiene servicio de internet
# internet.DeviceProtection = indica si tiene un servicio de protección de dispositivos, o si no tiene servicio de internet
# internet.TechSupport = indica si tiene un servicio de soporte tecnologico o si no tiene servicio de internet
# internet.StreamingTV = indica si tiene un servicio de Streaming de tv o si no tiene servicio de internet
# internet.StreamingMovies = indica si tiene un servicio de Streaming de peliculas o si no tiene servicio de internet
# account.Contract = indica el tipo de contrato (Un año, mes a mes, 2 años)
# account.PaperlessBilling = indica si el cliente recibe sus facturas en formato digital en lugar de papel
# account.PaymentMethod = indica como el cliente realiza el pago
# account.Charges.Monthly = indica el monto cobrado mes a mes al cliente
# account.Charges.Total = indica el total cobrado al cliente hasta el momento

print(df_clientes_normalizado)
print(df_clientes_normalizado.columns)
print(df_clientes_normalizado.info())
print('\n')

print(df_clientes_normalizado['Churn'].unique()) # notamos que la columna churn tiene registros vacios ('')
# churn tiene registros vacios

# REALIZAR NORMALIZACIÓN 

df_clientes_normalizado['customer.SeniorCitizen'] = df_clientes_normalizado['customer.SeniorCitizen'].astype(np.bool)

# al momento de cambiar el tipo de dato, notamos que hay 11 registros vacios en account.Charges.Total, se los reemplazará por nan
df_clientes_normalizado['account.Charges.Total'] = df_clientes_normalizado['account.Charges.Total'].replace(' ', np.nan).astype(np.float64)

# factura_digital
df_clientes_normalizado['account.PaperlessBilling'] = df_clientes_normalizado['account.PaperlessBilling'].replace('Yes', True).replace('No', False).astype(np.bool)

# tiene_pareja
df_clientes_normalizado['customer.Partner'] = df_clientes_normalizado['customer.Partner'].replace('Yes', True).replace('No', False).astype(np.bool)

# tiene_dependientes
df_clientes_normalizado['customer.Dependents'] = df_clientes_normalizado['customer.Dependents'].replace('Yes', True).replace('No', False).astype(np.bool)

# servicio_telefonico
df_clientes_normalizado['phone.PhoneService'] = df_clientes_normalizado['phone.PhoneService'].replace('Yes', True).replace('No', False).astype(np.bool)

# multiples_lineas
df_clientes_normalizado['phone.MultipleLines'] = df_clientes_normalizado['phone.MultipleLines'].replace('No phone service', False).replace('No', False).replace('Yes', True).astype(np.bool)

# servicio_seguridad
df_clientes_normalizado['internet.OnlineSecurity'] = df_clientes_normalizado['internet.OnlineSecurity'].replace('No internet service', False).replace('No', False).replace('Yes', True).astype(np.bool)

# servicio_backup
df_clientes_normalizado['internet.OnlineBackup'] = df_clientes_normalizado['internet.OnlineBackup'].replace('No internet service', False).replace('No', False).replace('Yes', True).astype(np.bool)

# servicio_proteccion
df_clientes_normalizado['internet.DeviceProtection'] = df_clientes_normalizado['internet.DeviceProtection'].replace('No internet service', False).replace('No', False).replace('Yes', True).astype(np.bool)

# servicio_soporte_tecnico
df_clientes_normalizado['internet.TechSupport'] = df_clientes_normalizado['internet.TechSupport'].replace('No internet service', False).replace('No', False).replace('Yes', True).astype(np.bool)

# streaming_tv
df_clientes_normalizado['internet.StreamingTV'] = df_clientes_normalizado['internet.StreamingTV'].replace('No internet service', False).replace('No', False).replace('Yes', True).astype(np.bool)

# streaming_peliculas
df_clientes_normalizado['internet.StreamingMovies'] = df_clientes_normalizado['internet.StreamingMovies'].replace('No internet service', False).replace('No', False).replace('Yes', True).astype(np.bool)

# reemplazamos en la columna churn del df normalizado todos los valores 'Yes' por True, asi mismo los valores 'No', False
# los registros de la columna churn que esten vacios ('') seran reemplazados por un nan
df_clientes_normalizado['Churn'] = df_clientes_normalizado['Churn'].replace('Yes', True).replace('No', False).replace('', np.nan)

print('\n')
print(df_clientes_normalizado['Churn'].sample(5))
print(df_clientes_normalizado['Churn'].isna().sum())
print(df_clientes_normalizado['Churn'].unique())


# en indicesNaN guardamos los indices de aquellos clientes que tengan un nan en su registro
indicesNaN = df_clientes_normalizado[df_clientes_normalizado['Churn'].isna()].index
indicesNaN = indicesNaN.append(df_clientes_normalizado[df_clientes_normalizado['account.Charges.Total'].isna()].index)

# en clientesNaN guardamos todos los registros del df que tengan algun valor NaN en las columnas, son 235 clientes
clientesNaN = df_clientes_normalizado.iloc[indicesNaN].copy()

# utilizamos el método dropna() que sirve para eliminar con valores NaN, con el parametro subset=['Churn'] indicamos que el filtrado se haga solamente evaluando la columna 'Churn'
# reseteamos los indices
df_clientes_normalizado = df_clientes_normalizado.dropna().reset_index(drop=True)

# cambiamos el tipo de dato de la columna 'Churn' a booleano
df_clientes_normalizado['Churn'] = df_clientes_normalizado['Churn'].astype(np.bool)

print(df_clientes_normalizado)
print('\n')

print(clientesNaN)
print('\n')


# cambiando el nombre de las columnas
df_clientes_normalizado = df_clientes_normalizado.rename(columns={'customerID': 'idCliente',
                                                                          'Churn': 'desercion_cliente',
                                                                          'customer.gender':'genero',
                                                                          'customer.SeniorCitizen': 'es_jubilado',
                                                                          'customer.Partner':'tiene_pareja',
                                                                          'customer.Dependents': 'tiene_dependientes',
                                                                          'customer.tenure': 'antiguedad',
                                                                          'phone.PhoneService' : 'servicio_telefonico',
                                                                          'phone.MultipleLines': 'multiples_lineas',
                                                                          'internet.InternetService': 'servicio_internet',
                                                                          'internet.OnlineSecurity': 'servicio_seguridad',
                                                                          'internet.OnlineBackup': 'servicio_backup',
                                                                          'internet.DeviceProtection': 'servicio_proteccion',
                                                                          'internet.TechSupport': 'servicio_soporte_tecnico',
                                                                          'internet.StreamingTV': 'streaming_tv',
                                                                          'internet.StreamingMovies': 'streaming_peliculas',
                                                                          'account.Contract': 'tipo_contrato',
                                                                          'account.PaperlessBilling': 'factura_digital',
                                                                          'account.PaymentMethod': 'metodo_pago',
                                                                          'account.Charges.Monthly': 'cargo_mensual',
                                                                          'account.Charges.Total': 'cargos_total'})

# traduciendo los tipos de contrato
df_clientes_normalizado['tipo_contrato'] = df_clientes_normalizado['tipo_contrato'].replace({'One year': 'un año',
                                                                                             'Month-to-month': 'mes a mes',
                                                                                             'Two year': 'dos años'})

# traduciendo los generos
df_clientes_normalizado['genero'] = df_clientes_normalizado['genero'].replace({'Female': 'femenino',
                                                                               'Male': 'masculino'})

# traduciendo los métodos de pago
df_clientes_normalizado['metodo_pago'] = df_clientes_normalizado['metodo_pago'].replace({'Mailed check': 'cheque por correo',
                                                                                         'Electronic check': 'cheque electrónico',
                                                                                         'Credit card (automatic)': 'tarjeta de crédito (automático)',
                                                                                         'Bank transfer (automatic)': 'transferencia bancaria (automática)'})

# traduciendo los servicios de internet
df_clientes_normalizado['servicio_internet'] = df_clientes_normalizado['servicio_internet'].replace({'Fiber optic': 'Fibra optica'})

# print('\n')
# print(df_clientes_normalizado.info())
# print('\n')
# print(df_clientes_normalizado['idCliente'].unique())
# print(df_clientes_normalizado['desercion_cliente'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['genero'].unique())
# print(df_clientes_normalizado['es_jubilado'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['tiene_pareja'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['tiene_dependientes'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['antiguedad'].unique())
# print(df_clientes_normalizado['servicio_telefonico'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['multiples_lineas'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['servicio_internet'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['servicio_seguridad'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['servicio_backup'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['servicio_proteccion'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['servicio_soporte_tecnico'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['streaming_tv'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['streaming_peliculas'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['tipo_contrato'].unique())
# print(df_clientes_normalizado['factura_digital'].unique()) # TIPO DE DATO CAMBIADO
# print(df_clientes_normalizado['metodo_pago'].unique())
# print(df_clientes_normalizado['cargo_mensual'].unique())
# print(df_clientes_normalizado['cargos_total'].unique()) # TIPO DE DATO CAMBIADO

# creando la columna cuentas diarias
df_clientes_normalizado['cuentas_diarias'] = round(df_clientes_normalizado['cargo_mensual']/30, 2)

print(df_clientes_normalizado)

# ANALISIS DESCRIPTIVO

columnas_analisis = ['antiguedad', 'cargo_mensual', 'cargos_total', 'cuentas_diarias']
print(df_clientes_normalizado[columnas_analisis].describe())

# DISTRIBUCION DE EVASIÓN

conteo_evasion = df_clientes_normalizado['desercion_cliente'].value_counts()
conteo_evasion_porcentajes = round(df_clientes_normalizado['desercion_cliente'].value_counts(normalize=True)*100, 2)

print(conteo_evasion)
print(conteo_evasion_porcentajes)

# POSIBLE GRAFICO: PIE

# grafico de tarta con la cantidad de clientes

sns.set_theme()

fig, axs = plt.subplots(1,2, figsize=(10, 6))
fig.suptitle('ANALISIS DE EVASIÓN DE CLIENTES', fontweight='bold', fontsize=16)

axs[0].pie(x=conteo_evasion, 
           labels=['No evadio', 'Evadio'], 
           startangle=90, 
           colors=['#47B39D',"#FDBDFD"],
           shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9}, 
           explode=(0, 0.1), 
           autopct=lambda pct: f'{int(round(pct/100 * sum(conteo_evasion)))}',
           textprops={'fontsize': 10, 'fontweight': 'bold'})

axs[0].set_title('Clientes que evadieron\n(Cantidad)', fontsize=11, fontweight='bold')

# grafico de tarta con los porcentajes
axs[1].pie(x=conteo_evasion_porcentajes, 
           labels=['No evadio', 'Evadio'], 
           startangle=90, 
           colors=['#47B39D',"#FDBDFD"],
           shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9}, 
           explode=(0, 0.1), 
           autopct=lambda pct: f'{pct:.1f}%', 
           textprops={'fontsize': 10, 'fontweight': 'bold'})

axs[1].set_title('Clientes que evadieron\n(Porcentaje)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('analisis_evasion_clientes.png', dpi=300)
plt.show()


# RECUENTO DE EVASION POR VARIABLES CATEGORICAS

conteo_evasion_genero = df_clientes_normalizado.groupby(['genero', 'desercion_cliente']).size().unstack()
conteo_evasion_contrato = df_clientes_normalizado.groupby(['tipo_contrato', 'desercion_cliente']).size().unstack()
conteo_evasion_pago = df_clientes_normalizado.groupby(['metodo_pago','desercion_cliente']).size().unstack()

conteo_evasion_genero.columns = ['No evadio', 'Evadio']

conteo_evasion_contrato.columns = ['No evadio', 'Evadio']

conteo_evasion_pago.columns = ['No evadio', 'Evadio']

print(conteo_evasion_genero)
print('\n')
print(conteo_evasion_contrato)
print('\n')
print(conteo_evasion_pago)


# POSIBLE GRAFICO: CATPLOT (VER REFERENCIA SEABORN)

fig, axs = plt.subplots(1, 3, figsize=(18,10), gridspec_kw={'width_ratios': [2, 2, 2]})
fig.subplots_adjust(wspace=0.3)
# con .patch nos posicionamos en el lienzo, con .set_facecolor(color) cambiamos el color del mismo
fig.patch.set_facecolor("#dad9d9")
fig.suptitle('ANALISIS DE EVASIÓN SEGÚN VARIABLES CATEGORICAS', ha='right', color="#3381c0", fontweight='bold')


conteo_evasion_genero.plot(kind='bar', 
                           ax=axs[0], 
                           color=['#47B39D','#462446'])
axs[0].legend(loc='best')
axs[0].set_ylim(top=axs[0].get_ylim()[1] * 1.15)
axs[0].set_title('Evasión por genero', loc='left', fontweight='bold')
axs[0].set_xlabel('')
axs[0].set_ylabel('')
axs[0].set_xticklabels(labels=conteo_evasion_genero.index, rotation=0)
axs[0].tick_params(left=False, labelleft=False)

for container in axs[0].containers:
    axs[0].bar_label(container, label_type='edge', padding=3, fontsize=10)


conteo_evasion_contrato.plot(kind='bar', 
                             ax=axs[1], 
                             color=['#47B39D','#462446'])

axs[1].legend(loc='best')
axs[1].set_ylim(top=axs[1].get_ylim()[1] * 1.15)
axs[1].set_title('Evasión por tipo de contrato', loc='left', fontweight='bold')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[1].set_xticklabels(labels=conteo_evasion_contrato.index, rotation=0)
axs[1].tick_params(left=False, labelleft=False)

for container in axs[1].containers:
    axs[1].bar_label(container, 
                     label_type='edge', 
                     padding=3, 
                     fontsize=10)

conteo_evasion_pago.plot(kind='bar', 
                         ax=axs[2], 
                         color=['#47B39D','#462446'])
axs[2].legend(loc='best')
axs[2].set_ylim(top=axs[2].get_ylim()[1] * 1.15)
axs[2].set_title('Evasión por método de pago', loc='left', fontweight='bold')
axs[2].set_xlabel('')
axs[2].set_ylabel('')
axs[2].set_xticklabels(labels=conteo_evasion_pago.index, rotation=20, ha='right')
axs[2].tick_params(left=False, labelleft=False)

for container in axs[2].containers:
    axs[2].bar_label(container, 
                     label_type='edge', 
                     padding=3, 
                     fontsize=10)
    
plt.savefig('evasion_variables_categoricas.png', dpi=300)
plt.show()

# CONTEO DE EVASION POR VARIABLES NUMERICAS

conteo_evasion_antiguedad = df_clientes_normalizado.groupby(['desercion_cliente', 'antiguedad']).size().unstack()

print('\n')
print(conteo_evasion_antiguedad)

# creamos una lista que almacenara valores entre 1 y 72 en tuplas dando pasos de 6 en 6
agrupaciones = [(i, i+5) for i in range(1, 73, 6)]
print(agrupaciones)

# creamos un df
df_conteo_antiguedad = pd.DataFrame()

for inicio, fin in agrupaciones:
    nombre_columna = f'{inicio}-{fin}'
    # creamos una columna nueva en el df, que recibira como valores la suma de aquellos registros que se encuentren en el rango especificado
    df_conteo_antiguedad[nombre_columna] = conteo_evasion_antiguedad.loc[:, inicio:fin].sum(axis=1)

df_conteo_antiguedad = df_conteo_antiguedad.T

print(df_conteo_antiguedad)

print(f'min: {df_conteo_antiguedad.min()}\nmax: {df_conteo_antiguedad.max()}')

fig, ax = plt.subplots(figsize=(14, 6))

df_conteo_antiguedad.plot(kind='bar', 
                          ax=ax, 
                          figsize=(14, 6))
ax.set_title('Deserción de Clientes por Intervalos de Antigüedad', loc='left', fontsize=18, fontweight='bold', pad=3)
ax.set_ylim(0, 1000)
ax.set_xlabel('Intervalor de antiguedad (meses)')
ax.set_ylabel('Cantidad de clientes')
ax.set_xticklabels(labels=df_conteo_antiguedad.index, rotation=0)
ax.legend(['No evadio', 'Evadio'],
          loc='upper center', 
          title='Deserción')

for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=2.5, fontsize=10)

plt.savefig('evasion_por_antiguedad.png', dpi=300)
plt.show()


# evasion por cargos totales
clientes_true = df_clientes_normalizado[df_clientes_normalizado['desercion_cliente'] == True]['cargos_total']
clientes_false = df_clientes_normalizado[df_clientes_normalizado['desercion_cliente'] == False]['cargos_total']

print(f'min clientes_true: {clientes_true.min()}\nmax clientes_true: {clientes_true.max()}\nmin clientes_false: {clientes_false.min()}\nmax clientes_false: {clientes_false.max()}\n')

print(clientes_true)

# creamos los intervalos
bins = range(0, 9000, 300) # cada intervalo es de 300 unidades

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist([clientes_false, clientes_true],
        bins=bins,
        label=['No evadio', 'Evadio'],
        stacked=False,
        alpha= 0.8)

ax.set_ylim(top=ax.get_ylim()[1] * 1.03)
ax.set_title('Deserción de Clientes según cargos totales', fontweight='bold')
ax.set_xlabel('Cargos totales')
ax.set_ylabel('Cantidad de clientes')
ax.legend()

plt.tight_layout()
plt.savefig('evasion_cargos_totales.png', dpi=300)
plt.show()