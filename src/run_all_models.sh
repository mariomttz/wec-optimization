#!/bin/bash

BOT_TOKEN="TELEGRAM_BOT_TOKEN"
CHAT_ID="TELEGRAM_CHAT_ID"
PROJECT_ROOT="/root/to/the/project"

declare -a DATASETS=("sydney_100" "perth_100" "perth_49" "sydney_49")
declare -a MODELS=("lr" "catboost" "nn" "svm")

send_telegram_message() {
    MESSAGE="$1"
    curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
        -d "chat_id=$CHAT_ID" \
        -d "text=$MESSAGE" > /dev/null
}

send_telegram_message "🤖 Iniciando la ejecución de los modelos en paralelo. ¡Te notificaré al terminar el proceso completo!"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        MODEL_PATH="models/$dataset/$model"
        SCRIPT_NAME="${dataset}_${model}.py"
        
        (
            cd "$PROJECT_ROOT/$MODEL_PATH" || { send_telegram_message "❌ Error: No se pudo navegar a $MODEL_PATH."; exit 1; }
            send_telegram_message "🛠️ Lanzando el modelo: $SCRIPT_NAME."
            python3 "$SCRIPT_NAME"
            
            if [ $? -eq 0 ]; then
            send_telegram_message "✅ El modelo $SCRIPT_NAME ha terminado su ejecución con éxito."
            
            else
            send_telegram_message "❌ Error: El modelo $SCRIPT_NAME falló durante la ejecución."
            
            fi
            
        ) &
    
    done
done

wait

cd "$PROJECT_ROOT"

echo "✅ Todos los modelos han sido ejecutados. Proceso finalizado."
send_telegram_message "🎉 Todos los modelos han sido ejecutados. Proceso finalizado."