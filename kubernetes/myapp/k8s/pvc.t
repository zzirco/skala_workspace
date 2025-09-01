apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${USER_NAME}-efs-sc-${SERVICE_NAME}-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteMany
  volumeMode: Filesystem
  resources:
    requests:
      storage: 10Mi
  storageClassName: efs-sc-shared
