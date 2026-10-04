// CI/CD on AWS: Jenkins (EC2, IAM role) -> Docker -> ECR -> EKS.  No AWS keys are stored in Jenkins.
pipeline {
  agent any
  options { timestamps(); disableConcurrentBuilds() }
  environment {
    AWS_REGION = 'ap-south-1'
    CLUSTER    = 'ecosort-eks'
    REPO       = 'ecosort-ai'
    TAG        = "${env.BUILD_NUMBER}"
    ECOSORT_BACKEND = 'demo'            // unit tests must not download the model
  }
  stages {
    stage('Setup')      { steps { sh 'python3 -m venv .venv && . .venv/bin/activate && pip install -q -r requirements.txt pytest ruff' } }
    stage('Lint')       { steps { sh '. .venv/bin/activate && ruff check src app training --select E9,F63,F7,F82' } }
    stage('Unit tests') { steps { sh '. .venv/bin/activate && pytest -q tests' } }
    stage('Data & model (DVC)') {
      when { expression { fileExists('.dvc/config') } }      // only once a DVC remote exists (give the Jenkins role S3 access if it is an S3 remote)
      steps { sh '. .venv/bin/activate && pip install -q -r requirements-train.txt && dvc pull && dvc repro' }
    }
    stage('Build & push to ECR') {
      steps { sh '''
        ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
        REG=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
        aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REG
        docker build -t $REG/$REPO:$TAG -t $REG/$REPO:latest .
        docker push $REG/$REPO:$TAG
        docker push $REG/$REPO:latest
        echo "$REG/$REPO:$TAG" > image.txt
      ''' }
    }
    stage('Deploy to EKS') {
      steps { sh '''
        aws eks update-kubeconfig --name $CLUSTER --region $AWS_REGION
        sed "s|IMAGE_PLACEHOLDER|$(cat image.txt)|" k8s/aws/ecosort-eks.yaml | kubectl apply -f -
        kubectl rollout status deployment/ecosort-ai --timeout=900s
        kubectl get pods,svc
      ''' }
    }
    stage('Smoke test') {
      steps { sh '''
        HOST=$(kubectl get svc ecosort-ai -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        echo "App URL: http://$HOST"
        for i in $(seq 1 30); do if curl -fsS "http://$HOST/_stcore/health"; then exit 0; fi; sleep 20; done
        echo "health check failed"; exit 1
      ''' }
    }
  }
  post { failure { echo 'Pipeline failed - check the stage above' } }
}
