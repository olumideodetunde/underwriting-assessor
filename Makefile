include .env
.EXPORT_ALL_VARIABLES:


APP_NAME=freq-mode-app
TAG=latest
TF_VAR_app_name=${APP_NAME}
REGISTRY_NAME=${APP_NAME}
TF_VAR_image=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REGISTRY_NAME}:${TAG}
TF_VAR_region=${AWS_REGION}
TF_VAR_model_bucket_name=${MODEL_BUCKET_NAME}
TF_VAR_load_model_path=${LOAD_MODEL_PATH}

setup-ecr:
	cd terraform/setup && terraform init && terraform apply -auto-approve

deploy-container:
	cd	src/app && sh deploy.sh

deploy-service:
	cd terraform/app && terraform init && terraform apply -auto-approve

destroy-ecr-and-service:
	cd terraform/app && terraform init && terraform destroy -auto-approve
	cd terraform/setup && terraform init && terraform destroy -auto-approve