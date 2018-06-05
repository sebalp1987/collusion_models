import requests
from utils import dateiterator
from datetime import datetime
import re
from bs4 import BeautifulSoup
import re

#source: http://www.omie.es/aplicaciones/datosftp/datosftp.jsp?path=/marginalpdbc/


proxies = {'http': 'http://10.243.241.44:8080', 'https':'10.243.241.44:8080'}

def precio_diario(start_date = '2007-07-01', end_date = '2015-01-01'):

    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()


    with open('file_precio.csv', 'w', newline='') as file:
        header = 'ANIO' + ';' + 'MES' + ';' + 'DIA' + ';' + 'HORA' + ';' + 'PESPANIA' + ';' + 'PPORTUGAL' + ';'
        file.write(header)
        for dt in dateiterator(start, end):
            dt = datetime.strftime(dt, '%Y%m%d')
            http = 'http://www.omie.es/datosPub/marginalpdbc/marginalpdbc_' + dt +'.1'
            page = requests.get(http)
            #print(page.status_code)  # Starting 2 is downloaded ok, Starting 4 not
            page = page.text
            if page.startswith('<!DOCTYPE'):
                http = 'http://www.omie.es/datosPub/marginalpdbc/marginalpdbc_' + dt + '.2'
                page = requests.get(http)
                # print(page.status_code)  # Starting 2 is downloaded ok, Starting 4 not
                page = page.text

            page = re.sub('\*', '', page)
            page = page.strip()
            page = re.sub('MARGINALPDBC;', '', page)

            file.write(page)
            print(dt)


def cantidad_diaria(start_date = '2007-07-01', end_date = '2015-01-01'):



    #http://www.omie.es/aplicaciones/datosftp/datosftp.jsp?path=/pdbc_stota/
    #EXTRAER:

    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()

    with open('file_produccion.csv', 'w', newline='') as file:

        header = 'ANIO' + ';' + 'MES' + ';' + 'DIA' + ';' + 'HORA' + ';' \
        + 'HIDRAULICA_CONVENC'+ ';' + 'HIDRAULICA_BOMBEO' + ';' + 'NUCLEAR' \
        + ';' + 'CARBON NACIONAL' + ';' + 'CARBON_IMPO' + ';' + 'CICLO_COMBINADO' \
        + ';' + 'FUEL_SIN_PRIMA' + ';' + 'FUEL_PRIMA' + ';' + 'REG_ESPECIAL' + '\n'

        file.write(header)

        for dt in dateiterator(start, end):
            dt = datetime.strftime(dt, '%Y%m%d')
            year = dt[0:4]
            month = dt[4:6]
            day = dt[6:8]

            http = 'http://www.omie.es/datosPub/pdbc_stota/pdbc_stota_' + dt + '.1'
            page = requests.get(http)
            page = page.text
            page_list = page.split(';')

            for i, iteri in enumerate(page_list):
                if iteri == '\r\n1':
                    hidr_conv = page_list[i + 3:i + 27]
                if iteri == '\r\n2':
                    hidr_bomb = page_list[i + 3:i + 27]
                if iteri == '\r\n3':
                    nuclear = page_list[i + 3:i + 27]
                if iteri == '\r\n4':
                    carbon_nac = page_list[i + 3:i + 27]
                if iteri == '\r\n5':
                    carbon_imp = page_list[i + 3:i + 27]
                if iteri == '\r\n6':
                    cc = page_list[i + 3:i + 27]
                if iteri == '\r\n7':
                    fuel_sprima = page_list[i + 3:i + 27]
                if iteri == '\r\n8':
                    fuel_cprima = page_list[i + 3:i + 27]
                if iteri == '\r\n9':
                    reg_esp = page_list[i + 3:i + 27]

            output = []
            for i in range(1,25,1):
                string = year + ';' + month + ';' + day + ';' + str(i) + ';' + hidr_conv[i-1] \
                    + ';' + hidr_bomb[i-1] + ';' + nuclear[i-1] + ';' + carbon_nac[i-1] \
                    + ';' + carbon_imp[i-1] + ';' + cc[i-1] + ';' + fuel_sprima[i-1] \
                    + ';' + fuel_cprima[i-1] + ';' + reg_esp[i-1]+'\n'
                file.write(string)
                print(dt)




def demanda_diaria(start_date = '2007-07-01', end_date = '2015-01-01'):



    #http://www.omie.es/aplicaciones/datosftp/datosftp.jsp?path=/pdbc_stota/
    #EXTRAER:

    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()

    with open('file_demanda.csv', 'w', newline='') as file:

        header = 'ANIO' + ';' + 'MES' + ';' + 'DIA' + ';' + 'HORA' + ';' \
                 + 'TOTAL_IMPORTACION_ES' + ';' + 'TOTAL_PRODUCCION_ES' + ';' + 'TOTAL_DEMANDA_NAC_ES' \
                 + ';' + 'TOTAL_EXPORTACIONES_ES' + ';' + 'TOTAL_DDA_ES' + \
                 ';' + 'TOTAL_POT_IND_ES' + ';' + 'TOTAL_PRODUCCION_POR' \
                 + ';' + 'TOTAL_DEMANDA_POR' + '\n'

        file.write(header)



        for dt in dateiterator(start, end):
            dt = datetime.strftime(dt, '%Y%m%d')
            year = dt[0:4]
            month = dt[4:6]
            day = dt[6:8]

            http = 'http://www.omie.es/datosPub/pdbc_stota/pdbc_stota_' + dt + '.1'
            page = requests.get(http)
            page = page.text
            page_list = page.split(';')
            print(page_list)

            for i, iteri in enumerate(page_list):
                if iteri == '\r\n44':
                    total_import = page_list[i + 3:i + 27]
                if iteri == '\r\n49':
                    total_prod = page_list[i + 3:i + 27]
                if iteri == '\r\n51':
                    total_dda_nac = page_list[i + 3:i + 27]
                if iteri == '\r\n53':
                    total_expo = page_list[i + 3:i + 27]
                if iteri == '\r\n59':
                    total_dda = page_list[i + 3:i + 27]
                if iteri == '\r\n64':
                    pot_indis = page_list[i + 3:i + 27]
                if iteri == '\r\n249':
                    prod_por = page_list[i + 3:i + 27]
                if iteri == '\r\n251':
                    dda_por = page_list[i + 3:i + 27]

            output = []
            for i in range(1, 25, 1):
                string = year + ';' + month + ';' + day + ';' + str(i) + ';' + total_import[i-1] \
                    + ';' + total_prod[i-1] + ';' + total_dda_nac[i-1] + ';' + total_expo[i-1] \
                    + ';' + total_dda[i-1] + ';' + pot_indis[i-1] + ';' + prod_por[i-1] + ';' + dda_por[i-1] + '\n'
                file.write(string)
                print(dt)


def energia_eolica():

    http = 'http://www.ree.es/es/balance-diario/peninsula/2012/01/01'
    page = requests.get(http)
    soup = BeautifulSoup(page.content, 'html.parser')

    otras= soup.find_all('tr', class_='datos odd')[3].get_text()
    solar_foto = soup.find_all('tr', class_='datos odd')[2].get_text()
    eolica = soup.find_all('tr', class_ = 'datos even')[2].get_text()
    solar_termica = soup.find_all('tr', class_='datos even')[3].get_text()
    otras = re.findall(r"[\w']+", otras)
    solar_foto = re.findall(r"[\w']+", solar_foto)
    eolica = re.findall(r"[\w']+", eolica)
    solar_termica = re.findall(r"[\w']+", solar_termica)

    print(solar_foto)
    print(otras)
    print(eolica)
    print(solar_termica)

energia_eolica()

# precio_diario()
# cantidad_diaria()
# demanda_diaria()









