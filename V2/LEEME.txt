Aquí se explica cómo ejecutar el programa de optimización.
Los requisitos son obtener la librería de optimización bayesopt y la carpeta con datos de las noches con sus contenidos organizados de la siguiente forma:

bayesopt							<- En caso de no tener bayesopt instalado como librería también es posible tener la carpeta de la librería aquí
...
Data							<- La carpeta raíz de las grabaciones
	Data/*_AR1_0*				<- Ejemplo de carpeta raíz de la grabación de un paciente en una noche concreta
		Data/*_AR1_0/*.bbt		<- Los archivos
		Data/*_AR1_0/*.bbt.p	<- de la grabación
...
optimizer.py						<- El programa python que hace la optimización
...
launcher.sh							<- Un script para facilitar la optimización de todas las noches en la carpeta raíz (en este caso Data)

El programa de python una vez cumplidos los requisitos se puede ejecutar de la siguiente forma (ajustando los nombres a las noches disponibles):
python optimizer.py -f AR1_0 -d Data -dbd -p

o por ejemplo para además especificar un tramo concreto a explorar dentro de una noche:
python optimizer.py -f AR5_0 -d Data -start 0 -end 3273712 -dbd -p

El optimizador tiene muchas otras opciones que se pueden revisar al final del fichero .py o a continuación:

OBLIGATORIOS:
"-f" o "--file" para identificar el fichero a optimizar (e.g. '_AR1_0').
"-d" o "--directory" para especificar el directorio con el fichero a optimizar (e.g. 'Data')
OPCIONALES:
"-a" o "--alphas" para sobrescribir coeficientes para la suma pesada. Se requieren 4: CMAE, CSD, PUP y PnotUP. Por defecto son: '0.1','0.0','1.0','1.0'.
"-lb" o "--lowbounds" para sobrescribir los límites inferiores del espacio de exploración. Se requieren 3: th_max, k_pll y f_nco. Por defecto son: '-80','0.1','0.8'.
"-dv" o "--defval" para cambiar los valores por defecto de los parámetros. Se requieren 3, igual que lb. Por defecto son: '-40','0.1','1.5'.
"-ub" o "--upbounds"  para cambiar los límites superiores del espacio de exploración. Se requieren 3, igual que lb. Por defecto son: '-20','1.0','10'.
"-op" o "--optmask" para especificar qué parámetros se optimizan. Se requieren 3, igual que lb. 0 para usar el valor por defecto y 1 o más para optimizar. Por defecto todos se optimizan.
"-start" o "--start_sample" para especificar el instante de la grabación a partir del cual se extraen datos (por defecto el principio). Tiene que pertenecer a la noche y ser menor que -end (e.g. 955167).
"-end" o "--end_sample" para especificar el instante de la grabación a partir del cual se deja de extraer datos (por defecto el final). Tiene que pertenecer a la noche y ser mayor que -start (e.g. 1460208).
"-in" o "--init" son las teraciones para la construcción del modelo inicial (de exploración, no optimización). Por defecto 16.
"-it" o "--iter" son las teraciones para el proceso de optimización. Por defecto 32.
"-fj" o "--force_jump" para especificar el número de iteraciones sin mejora antes de dar un salto aleatorio. Por defecto 2.
"-ns" o "--noise" para modificar el ruido aleatorio aplicado a las muestras. Por defecto 0.0001.
"-dbd" o "--divbydef" para dividir métricas por valor por defecto (se recomienda usarlo).
"-p" o "--plot" para dibujar métricas en los puntos explorados al final de la optimización.

Se añade además un script para facilitar la optimización de todas las noches en la carpeta raíz, pero está configurado para la carpeta Data
y las seis noches con las que se han hecho las pruebas. Para adaptarlo a cualquier otro set de pruebas es tan sencillo como cambiar dentro del script 4 variables.

ar_directory="Data" cambiando "Data" por el nombre de la carpeta de datos.
ar_files=("_AR1_0" "_AR2_0" "_AR3_0" "_AR4_0" "_AR5_0" "_AR6_0") cambiando los nombres por los de las noches que se tengan o agregando más (basta con poner la parte del nombre que no se repita entre archivos).
ar_start_params y ar_end_params tienen tantas filas como archivos de grabaciones haya, y en cada fila están los comienzos y los finales que se quieren analizar para cada grabación.
Cada comienzo tiene que ir ligado de un final en la fila y la posición correspondiente, y viceversa.

Se puede utilizar el script launcher.sh de la siguiente forma, siendo todos los parámetros necesarios:
./launcher.sh -n 1 -w Y -p N

-n denota qué noche analizar (el orden lo denota el orden en que se han puesto los nombres de archivos en la variable ar_files).
-w a 'yes' o 'y' para analizar la noche entera y a 'no' o 'n' para no hacerlo.
-p a 'yes' o 'y' para analizar los trozos de noche especificados y a 'no' o 'n' para no hacerlo.

Una vez que se especifican qué optimizaciones hacer se ejecutarán una por una de forma secuencial.
Cada optimización sacará unos archivos de logs cuyos nombres especifican el nombre del archivo, el trozo optimizado y el momento en que finalizó.