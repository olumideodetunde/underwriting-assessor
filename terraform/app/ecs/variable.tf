variable "app_name" {
  description = "Name of the app."
  type        = string
}
variable "region" {
  description = "AWS region to deploy the network to."
  type        = string
}
variable "image" {
  description = "Image used to start the container. Should be in repository-url/image:tag format."
  type        = string
}
variable "vpc_id" {
  description = "ID of the VPC where the ECS will be hosted."
  type        = string
}
variable "public_subnet_ids" {
  description = "IDs of public subnets where the ALB will be attached to."
  type        = list(string)
}
variable "private_subnet_ids" {
  description = "IDs of private subnets where the ECS service will be deployed to."
  type        = list(string)
}

variable "model_bucket_name" {
  description = "s3 bucker name where model is stored"
  type = string
}

variable "load_model_path" {
  description = "s3 path where model is stored"
  type = string
}