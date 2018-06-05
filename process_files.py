import pandas as pd
import STRING
import numpy as np

# FALTA EOLICA

def process_date():
    df_demanda = pd.read_csv(STRING.demanda, sep=';', dtype={'ANIO': str, 'MES': str, 'DIA':str})
    bad_columns = ['TOTAL_IMPORTACION_ES', 'TOTAL_PRODUCCION_ES', 'TOTAL_DEMANDA_NAC_ES', 'TOTAL_EXPORTACIONES_ES',
                   'TOTAL_DDA_ES', 'TOTAL_POT_IND_ES', 'TOTAL_PRODUCCION_POR', 'TOTAL_DEMANDA_POR']
    for i in bad_columns:
        df_demanda[i] = df_demanda[i].str.replace('.','')
        df_demanda[i] = df_demanda[i].str.replace(',','.')
        df_demanda[i] = df_demanda[i].map(float)

    df_precio = pd.read_csv(STRING.precio, sep=';', dtype={'ANIO': str, 'MES': str, 'DIA':str})

    df_produccion = pd.read_csv(STRING.output, sep=';', dtype={'ANIO': str, 'MES': str, 'DIA':str})
    bad_columns = ['HIDRAULICA_CONVENC', 'HIDRAULICA_BOMBEO', 'NUCLEAR', 'CARBON NACIONAL',
                   'CARBON_IMPO', 'CICLO_COMBINADO',
                   'FUEL_SIN_PRIMA', 'FUEL_PRIMA', 'REG_ESPECIAL']
    for i in bad_columns:
        df_produccion[i] = df_produccion[i].str.replace('.','')
        df_produccion[i] = df_produccion[i].str.replace(',','.')
        df_produccion[i] = df_produccion[i].map(float)
    
    df_fecha_subasta = pd.read_csv(STRING.subastas_fecha, sep=';', dtype={'FECHA':str})
    del df_precio['Unnamed: 6']

    df_demanda['FECHA'] = df_demanda['DIA'].map(str) + df_demanda['MES'].map(str) + df_demanda['ANIO'].map(str)
    df_produccion['FECHA'] =df_produccion['DIA'].map(str) + df_produccion['MES'].map(str)+  df_produccion['ANIO'].map(str)
    df_precio['FECHA'] = df_precio['DIA'].map(str) + df_precio['MES'].map(str) +  df_precio['ANIO'].map(str)

    df_fecha_subasta['FECHA'] = df_fecha_subasta['FECHA'].map(str)

    df_demanda['FECHA_HORA'] = df_demanda['FECHA'].map(str) + df_demanda['HORA'].map(str)
    df_precio['FECHA_HORA'] = df_precio['FECHA'].map(str) + df_precio['HORA'].map(str)
    df_produccion['FECHA_HORA'] = df_produccion['FECHA'].map(str) + df_produccion['HORA'].map(str)

    df = pd.merge(df_precio, df_fecha_subasta, how='left', on='FECHA', suffixes=('',''))

    delete_var = ['ANIO', 'MES', 'DIA', 'HORA', 'FECHA']
    for i in delete_var:
        del df_demanda[i]
        del df_produccion[i]

    df = pd.merge(df, df_demanda, how='left', on='FECHA_HORA')
    df = pd.merge(df, df_produccion, how='left', on='FECHA_HORA')

    del df_demanda
    del df_fecha_subasta
    del df_precio
    del df_produccion

    df = df.fillna(0)
    df.to_csv('processed_file.csv', sep=';', index=False)


def daily_var_file():
    df_daily = pd.read_csv(STRING.daily, sep=';', dtype={'FECHA': str})
    df_daily = df_daily.interpolate(limit_direction='backward', method='nearest')
    df_daily['FECHA'] = df_daily['FECHA'].str.replace('/', '')

    df = pd.read_csv(STRING.processed_file, sep=';', dtype={'FECHA': str})
    df = pd.merge(df, df_daily, how='left', on='FECHA')

    df.to_csv('processed_merge_file.csv', sep=';', index=False)

def generate_dummy():
    df = pd.read_csv(STRING.processed_merge, sep=';')

    df.loc[df['DUMMY'] == 0, 'DUMMY'] = np.NaN

    dummy_values = [5, 10, 15, 20, 30]
    for i in dummy_values:
        name = 'DUMMY_' + str(i) + '_DAY'
        df[name] = pd.Series(df['DUMMY'], index=df.index)
        rows = i*24
        df[name] = df[name].interpolate(limit=rows, limit_direction='backward', method='values')
        df[name] = df[name].fillna(0)

    df['DUMMY'] = df['DUMMY'].fillna(0)
    df = df.dropna(axis=0, how='any')

    df.to_csv('final.csv', sep=';', index=False)
    df['DUMMY'] = df['DUMMY'].interpolate()


if __name__ == '__main__':
    process_date()
    daily_var_file()
    generate_dummy()
