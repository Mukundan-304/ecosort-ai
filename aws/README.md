# EcoSort AI on AWS (Jenkins on EC2 -> Docker -> ECR -> EKS)
Region ap-south-1. Replace <you>, <vpc>, <subnet>, <ip>. Delete everything when done (see Teardown) - EKS + EC2 + load balancer cost money.
```bat
aws configure                                   & aws sts get-caller-identity
aws ecr create-repository --repository-name ecosort-ai --region ap-south-1 --image-scanning-configuration scanOnPush=true
eksctl create cluster -f aws/eks-cluster.yaml   & kubectl get nodes
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text --region ap-south-1
aws ec2 describe-subnets --filters Name=vpc-id,Values=<vpc> --query "Subnets[0].SubnetId" --output text --region ap-south-1
curl -s https://checkip.amazonaws.com
aws cloudformation deploy --template-file aws/jenkins-ec2.yaml --stack-name ecosort-jenkins --capabilities CAPABILITY_IAM --region ap-south-1 --parameter-overrides VpcId=<vpc> SubnetId=<subnet> MyIpCidr=<ip>/32 RepoUrl=https://github.com/<you>/ecosort-ai.git
aws cloudformation describe-stacks --stack-name ecosort-jenkins --region ap-south-1 --query "Stacks[0].Outputs"
aws eks create-access-entry --cluster-name ecosort-eks --principal-arn <JenkinsRoleArn> --region ap-south-1
aws eks associate-access-policy --cluster-name ecosort-eks --principal-arn <JenkinsRoleArn> --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy --access-scope type=cluster --region ap-south-1
```
Jenkins password: EC2 console > Connect > Session Manager, then `sudo docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword`.
Create a Pipeline job (Pipeline script from SCM, your repo, `Jenkinsfile`) and Build Now. App URL = `kubectl get svc ecosort-ai`.
Teardown: `kubectl delete -f k8s/aws/ecosort-eks.yaml` -> `eksctl delete cluster -f aws/eks-cluster.yaml` -> `aws cloudformation delete-stack --stack-name ecosort-jenkins --region ap-south-1` -> `aws ecr delete-repository --repository-name ecosort-ai --force --region ap-south-1`
