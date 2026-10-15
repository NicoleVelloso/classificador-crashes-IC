set -e

if [ -z "$VIRTUAL_ENV" ]; then
    echo "AVISO: o ambiente virtual nao parece estar ativo."
    echo "Rode primeiro:  source venv/bin/activate"
    echo
    if [ -t 0 ]; then
        read -p "Continuar mesmo assim? (s/N) " resposta
        if [ "$resposta" != "s" ] && [ "$resposta" != "S" ]; then
            exit 1
        fi
    else
        echo "Entrada nao interativa: seguindo mesmo assim."
        echo "Se faltar biblioteca, o erro aparece no primeiro script."
        echo
    fi
fi

echo
echo "#############################################################"
echo "#  1/5  CARREGAMENTO E VERIFICACAO DO DATASET"
echo "#############################################################"
python3 01_carregar_dados.py

echo
echo "#############################################################"
echo "#  2/5  TREINAMENTO DO MODELO"
echo "#############################################################"
python3 02_treinar_modelo.py

echo
echo "#############################################################"
echo "#  3/5  AVALIACAO COM VALIDACAO CRUZADA"
echo "#############################################################"
python3 03_avaliar_modelo.py

echo
echo "#############################################################"
echo "#  4/5  CLASSIFICACAO DOS CRASHES NOVOS"
echo "#############################################################"
python3 04_classificar_crashes_novos.py

echo
echo "#############################################################"
echo "#  5/5  GERACAO DO DASHBOARD"
echo "#############################################################"
python3 05_gerar_dashboard.py

echo
echo "#############################################################"
echo "#  CONCLUIDO"
echo "#############################################################"
echo
echo "Diagnostico opcional: python3 06_diagnostico_features.py"
echo
