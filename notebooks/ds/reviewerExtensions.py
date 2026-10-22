from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


CLASS_NAMES = ('AK', 'BCC', 'MEL', 'NV', 'SCC')
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


def findProjectRoot(startPath: Path) -> Path:
    for candidateRoot in (startPath, *startPath.parents):
        if (candidateRoot / 'data' / 'processed').exists() and (candidateRoot / 'notebooks').exists():
            return candidateRoot
    raise RuntimeError('Run from the skin-domain-shift project or one of its subdirectories.')


def setRandomSeed(randomSeed: int) -> None:
    random.seed(randomSeed)
    np.random.seed(randomSeed)
    torch.manual_seed(randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(randomSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fromProjectPath(projectRoot: Path, pathValue: str) -> Path:
    return (projectRoot / Path(pathValue)).resolve()


def manifestSha256(pathValue: Path) -> str:
    return hashlib.sha256(pathValue.read_bytes()).hexdigest()


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, leftValue: str, rightValue: str) -> None:
        leftRoot = self.find(leftValue)
        rightRoot = self.find(rightValue)
        if leftRoot != rightRoot:
            smallerRoot, largerRoot = sorted((leftRoot, rightRoot))
            self.parent[largerRoot] = smallerRoot


def prepareIsicDevelopmentSplit(projectRoot: Path) -> dict:
    processedRoot = projectRoot / 'data' / 'processed'
    sourcePath = processedRoot / 'isicTrain.csv'
    duplicatePath = processedRoot / 'duplicateGroups.csv'
    sourceFrame = pd.read_csv(sourcePath, low_memory=False)
    if not sourceFrame['officialPartition'].eq('officialTrain').all():
        raise ValueError('ISIC development data must come only from the official training partition.')
    if set(sourceFrame['harmonizedLabel']) != set(CLASS_NAMES):
        raise ValueError('ISIC development data does not contain the expected five classes.')

    imageTokens = [f'image:{value}' for value in sourceFrame['imageId'].astype(str)]
    groupBuilder = UnionFind(imageTokens)
    tokenByImage = dict(zip(sourceFrame['imageId'].astype(str), imageTokens))
    for _, lesionRows in sourceFrame.dropna(subset=['lesionId']).groupby('lesionId'):
        lesionTokens = [tokenByImage[value] for value in lesionRows['imageId'].astype(str)]
        for token in lesionTokens[1:]:
            groupBuilder.union(lesionTokens[0], token)

    if duplicatePath.exists():
        duplicateFrame = pd.read_csv(duplicatePath, low_memory=False)
        pathToImage = dict(zip(sourceFrame['imagePath'].astype(str), sourceFrame['imageId'].astype(str)))
        duplicateFrame = duplicateFrame.loc[duplicateFrame['imagePath'].astype(str).isin(pathToImage)].copy()
        for _, duplicateRows in duplicateFrame.groupby('duplicateGroup'):
            duplicateTokens = [tokenByImage[pathToImage[pathValue]] for pathValue in duplicateRows['imagePath'].astype(str)]
            for token in duplicateTokens[1:]:
                groupBuilder.union(duplicateTokens[0], token)

    sourceFrame['developmentGroup'] = [groupBuilder.find(token) for token in imageTokens]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = []
    globalDistribution = sourceFrame['harmonizedLabel'].value_counts(normalize=True).reindex(CLASS_NAMES).fillna(0)
    for foldNumber, (trainIndices, valIndices) in enumerate(
        splitter.split(sourceFrame, sourceFrame['harmonizedLabel'], groups=sourceFrame['developmentGroup']), start=1,
    ):
        valFrame = sourceFrame.iloc[valIndices]
        valDistribution = valFrame['harmonizedLabel'].value_counts(normalize=True).reindex(CLASS_NAMES).fillna(0)
        distributionError = float((valDistribution - globalDistribution).abs().sum())
        sizeError = abs(len(valFrame) / len(sourceFrame) - 0.20)
        candidates.append((distributionError + sizeError, foldNumber, trainIndices, valIndices))
    _, selectedFold, trainIndices, valIndices = min(candidates, key=lambda item: item[0])
    trainFrame = sourceFrame.iloc[trainIndices].copy()
    valFrame = sourceFrame.iloc[valIndices].copy()
    trainFrame['split'] = 'isicDevelopmentTrain'
    valFrame['split'] = 'isicDevelopmentVal'

    lesionOverlap = set(trainFrame['lesionId'].dropna().astype(str)) & set(valFrame['lesionId'].dropna().astype(str))
    groupOverlap = set(trainFrame['developmentGroup']) & set(valFrame['developmentGroup'])
    assert not lesionOverlap and not groupOverlap
    assert len(trainFrame) + len(valFrame) == len(sourceFrame)

    trainPath = processedRoot / 'isicDevelopmentTrain.csv'
    valPath = processedRoot / 'isicDevelopmentVal.csv'
    reportPath = processedRoot / 'isicDevelopmentSplitReport.json'
    trainFrame.to_csv(trainPath, index=False)
    valFrame.to_csv(valPath, index=False)
    report = {
        'randomSeed': 42,
        'selectedFold': selectedFold,
        'grouping': 'connected lesionId and confirmed exact-duplicate groups',
        'trainImageCount': len(trainFrame),
        'valImageCount': len(valFrame),
        'trainGroupCount': int(trainFrame['developmentGroup'].nunique()),
        'valGroupCount': int(valFrame['developmentGroup'].nunique()),
        'lesionOverlapCount': len(lesionOverlap),
        'connectedGroupOverlapCount': len(groupOverlap),
        'trainClassCounts': trainFrame['harmonizedLabel'].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict(),
        'valClassCounts': valFrame['harmonizedLabel'].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict(),
        'officialTestUsedForSplit': False,
    }
    reportPath.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


class SkinDataset(Dataset):
    def __init__(self, projectRoot: Path, frame: pd.DataFrame, transformValue):
        self.projectRoot = projectRoot
        self.frame = frame.reset_index(drop=True).copy()
        self.transformValue = transformValue
        self.classToIndex = {className: index for index, className in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(fromProjectPath(self.projectRoot, row['imagePath'])) as image:
            imageTensor = self.transformValue(image.convert('RGB'))
        return imageTensor, self.classToIndex[row['harmonizedLabel']], row['imageId']


def buildTransforms(imageSize: int):
    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop(imageSize, scale=(0.80, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
    ])
    evaluationTransform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
    ])
    return trainTransform, evaluationTransform


def createModel(modelName: str, device: torch.device):
    if modelName == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    elif modelName == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    else:
        raise ValueError(f'Unsupported model: {modelName}')
    return model.to(device)


def forwardFeaturesAndLogits(model: nn.Module, images: torch.Tensor):
    features = model.features(images)
    features = model.avgpool(features)
    features = torch.flatten(features, 1)
    logits = model.classifier(features)
    return features, logits


def coralLoss(sourceFeatures: torch.Tensor, targetFeatures: torch.Tensor) -> torch.Tensor:
    featureCount = sourceFeatures.shape[1]
    sourceCentered = sourceFeatures - sourceFeatures.mean(dim=0, keepdim=True)
    targetCentered = targetFeatures - targetFeatures.mean(dim=0, keepdim=True)
    sourceCovariance = sourceCentered.T @ sourceCentered / max(sourceFeatures.shape[0] - 1, 1)
    targetCovariance = targetCentered.T @ targetCentered / max(targetFeatures.shape[0] - 1, 1)
    return (sourceCovariance - targetCovariance).pow(2).sum() / (4.0 * featureCount * featureCount)


def setFeatureTraining(model: nn.Module, trainFeatures: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainFeatures
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def calculateMetrics(predictions: pd.DataFrame) -> dict:
    trueLabels = predictions['trueLabel']
    predictedLabels = predictions['predictedLabel']
    recallValues = recall_score(trueLabels, predictedLabels, labels=list(CLASS_NAMES), average=None, zero_division=0)
    return {
        'sampleCount': len(predictions),
        'macroF1': float(f1_score(trueLabels, predictedLabels, labels=list(CLASS_NAMES), average='macro', zero_division=0)),
        'balancedAccuracy': float(balanced_accuracy_score(trueLabels, predictedLabels)),
        'perClassRecall': {className: float(value) for className, value in zip(CLASS_NAMES, recallValues)},
        'confusionMatrix': confusion_matrix(trueLabels, predictedLabels, labels=list(CLASS_NAMES)).tolist(),
    }


def bootstrapIntervals(predictions: pd.DataFrame, randomSeed: int, bootstrapCount: int) -> dict:
    randomGenerator = np.random.default_rng(randomSeed)
    classIndices = {
        className: predictions.index[predictions['trueLabel'].eq(className)].to_numpy()
        for className in CLASS_NAMES
    }
    bootstrapRows = []
    for _ in range(bootstrapCount):
        sampledIndices = np.concatenate([
            randomGenerator.choice(indices, size=len(indices), replace=True)
            for indices in classIndices.values()
        ])
        sampled = predictions.loc[sampledIndices]
        bootstrapRows.append(calculateMetrics(sampled))
    return {
        metricName: {
            'lower95': float(np.quantile([row[metricName] for row in bootstrapRows], 0.025)),
            'upper95': float(np.quantile([row[metricName] for row in bootstrapRows], 0.975)),
        }
        for metricName in ('macroF1', 'balancedAccuracy')
    }


@torch.inference_mode()
def predictFrame(model: nn.Module, dataLoader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for images, labels, imageIds in tqdm(dataLoader, desc='Evaluating'):
        images = images.to(device, non_blocking=True)
        ampContext = torch.amp.autocast('cuda', enabled=device.type == 'cuda') if device.type == 'cuda' else nullcontext()
        with ampContext:
            _, logits = forwardFeaturesAndLogits(model, images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictedIndices = probabilities.argmax(axis=1)
        for rowIndex, imageId in enumerate(imageIds):
            row = {
                'imageId': imageId,
                'trueLabel': CLASS_NAMES[int(labels[rowIndex])],
                'predictedLabel': CLASS_NAMES[int(predictedIndices[rowIndex])],
            }
            row.update({f'probability{className}': float(probabilities[rowIndex, classIndex]) for classIndex, className in enumerate(CLASS_NAMES)})
            rows.append(row)
    return pd.DataFrame(rows)


def smokeSample(frame: pd.DataFrame, rowCount: int) -> pd.DataFrame:
    return frame.groupby('harmonizedLabel', group_keys=False).head(rowCount).reset_index(drop=True)


def runExtensionExperiment(projectRoot: Path, experimentName: str, runMode: str, randomSeed: int) -> dict:
    if experimentName not in {'efficientnet', 'reverse', 'coral'}:
        raise ValueError('Experiment must be efficientnet, reverse, or coral.')
    if runMode not in {'smoke', 'full'}:
        raise ValueError('Run mode must be smoke or full.')
    setRandomSeed(randomSeed)
    prepareIsicDevelopmentSplit(projectRoot)
    processedRoot = projectRoot / 'data' / 'processed'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imageSize = int(os.environ.get('SKIN_IMAGE_SIZE', '224'))
    batchSize = int(os.environ.get('SKIN_BATCH_SIZE', '32' if runMode == 'full' else '8'))
    evaluationBatchSize = int(os.environ.get('SKIN_EVAL_BATCH_SIZE', '64' if device.type == 'cuda' else '16'))
    warmupEpochs = int(os.environ.get('SKIN_WARMUP_EPOCHS', '3' if runMode == 'full' else '1'))
    fineTuneEpochs = int(os.environ.get('SKIN_FINETUNE_EPOCHS', '12' if runMode == 'full' else '0'))
    patience = int(os.environ.get('SKIN_PATIENCE', '4'))
    coralWeight = float(os.environ.get('SKIN_CORAL_WEIGHT', '0.1'))
    modelName = 'efficientnet_b0' if experimentName == 'efficientnet' else 'mobilenet_v3_small'
    if runMode == 'smoke' and experimentName == 'coral':
        fineTuneEpochs = max(fineTuneEpochs, 1)

    if experimentName == 'reverse':
        trainPath = processedRoot / 'isicDevelopmentTrain.csv'
        valPath = processedRoot / 'isicDevelopmentVal.csv'
        sourceTestPath = processedRoot / 'isicTest.csv'
        externalTestPath = processedRoot / 'clinicalTest.csv'
        sourceDomain = 'isic'
        externalDomain = 'clinical'
    else:
        trainPath = processedRoot / 'clinicalTrain.csv'
        valPath = processedRoot / 'clinicalVal.csv'
        sourceTestPath = processedRoot / 'clinicalTest.csv'
        externalTestPath = processedRoot / 'isicTest.csv'
        sourceDomain = 'clinical'
        externalDomain = 'isic'

    trainFrame = pd.read_csv(trainPath, low_memory=False)
    valFrame = pd.read_csv(valPath, low_memory=False)
    sourceTestFrame = pd.read_csv(sourceTestPath, low_memory=False)
    externalTestFrame = pd.read_csv(externalTestPath, low_memory=False)
    targetFrame = pd.read_csv(processedRoot / 'isicDevelopmentTrain.csv', low_memory=False) if experimentName == 'coral' else None
    if runMode == 'smoke':
        trainFrame = smokeSample(trainFrame, 4)
        valFrame = smokeSample(valFrame, 2)
        sourceTestFrame = smokeSample(sourceTestFrame, 3)
        externalTestFrame = smokeSample(externalTestFrame, 3)
        if targetFrame is not None:
            targetFrame = smokeSample(targetFrame, 4)

    trainTransform, evaluationTransform = buildTransforms(imageSize)
    generator = torch.Generator().manual_seed(randomSeed)
    trainLoader = DataLoader(SkinDataset(projectRoot, trainFrame, trainTransform), batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda', generator=generator)
    valLoader = DataLoader(SkinDataset(projectRoot, valFrame, evaluationTransform), batch_size=evaluationBatchSize, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    sourceTestLoader = DataLoader(SkinDataset(projectRoot, sourceTestFrame, evaluationTransform), batch_size=evaluationBatchSize, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    externalTestLoader = DataLoader(SkinDataset(projectRoot, externalTestFrame, evaluationTransform), batch_size=evaluationBatchSize, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    targetLoader = None
    if targetFrame is not None:
        targetLoader = DataLoader(SkinDataset(projectRoot, targetFrame, trainTransform), batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(randomSeed + 1), drop_last=len(targetFrame) >= batchSize)

    model = createModel(modelName, device)
    classCounts = trainFrame['harmonizedLabel'].value_counts().reindex(CLASS_NAMES)
    classWeights = len(trainFrame) / (len(CLASS_NAMES) * classCounts)
    lossFunction = nn.CrossEntropyLoss(weight=torch.tensor(classWeights.values, dtype=torch.float32, device=device))
    gradientScaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    def evaluateValidation() -> dict:
        predictions = predictFrame(model, valLoader, device)
        return calculateMetrics(predictions)

    def trainEpoch(optimizer, useCoral: bool) -> tuple[float, float]:
        model.train()
        targetIterator = iter(targetLoader) if useCoral and targetLoader is not None else None
        totalClassificationLoss = 0.0
        totalCoralLoss = 0.0
        totalRows = 0
        for sourceImages, sourceLabels, _ in tqdm(trainLoader, desc='Training'):
            sourceImages = sourceImages.to(device, non_blocking=True)
            sourceLabels = sourceLabels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ampContext = torch.amp.autocast('cuda', enabled=device.type == 'cuda') if device.type == 'cuda' else nullcontext()
            with ampContext:
                sourceFeatures, sourceLogits = forwardFeaturesAndLogits(model, sourceImages)
                classificationLoss = lossFunction(sourceLogits, sourceLabels)
                alignmentLoss = torch.zeros((), device=device)
                if targetIterator is not None:
                    try:
                        targetImages, _, _ = next(targetIterator)
                    except StopIteration:
                        targetIterator = iter(targetLoader)
                        targetImages, _, _ = next(targetIterator)
                    targetImages = targetImages.to(device, non_blocking=True)
                    targetFeatures, _ = forwardFeaturesAndLogits(model, targetImages)
                    alignmentLoss = coralLoss(sourceFeatures, targetFeatures)
                totalLoss = classificationLoss + coralWeight * alignmentLoss
            gradientScaler.scale(totalLoss).backward()
            gradientScaler.step(optimizer)
            gradientScaler.update()
            rowCount = len(sourceLabels)
            totalRows += rowCount
            totalClassificationLoss += float(classificationLoss.detach()) * rowCount
            totalCoralLoss += float(alignmentLoss.detach()) * rowCount
        return totalClassificationLoss / totalRows, totalCoralLoss / totalRows

    artifactRoot = projectRoot / 'artifacts' / 'extensions' / runMode / experimentName / f'seed{randomSeed}'
    figureRoot = projectRoot / 'reports' / 'figures' / 'extensions' / runMode / experimentName / f'seed{randomSeed}'
    artifactRoot.mkdir(parents=True, exist_ok=True)
    figureRoot.mkdir(parents=True, exist_ok=True)
    checkpointPath = artifactRoot / 'bestModel.pt'
    historyRows = []
    bestValMacroF1 = -1.0
    epochsWithoutImprovement = 0
    epochNumber = 0
    stopTraining = False
    startTime = time.perf_counter()
    phases = [
        ('classifierWarmup', warmupEpochs, False, 1e-3),
        ('fullFineTune', fineTuneEpochs, True, 1e-4),
    ]
    for phaseName, epochCount, trainFeatures, learningRate in phases:
        if epochCount <= 0 or stopTraining:
            continue
        setFeatureTraining(model, trainFeatures)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learningRate, weight_decay=1e-4)
        for _ in range(epochCount):
            epochNumber += 1
            classificationLoss, alignmentLoss = trainEpoch(optimizer, experimentName == 'coral' and trainFeatures)
            valMetrics = evaluateValidation()
            historyRows.append({
                'epoch': epochNumber,
                'phase': phaseName,
                'classificationLoss': classificationLoss,
                'coralLoss': alignmentLoss,
                'valMacroF1': valMetrics['macroF1'],
                'valBalancedAccuracy': valMetrics['balancedAccuracy'],
            })
            if valMetrics['macroF1'] > bestValMacroF1 + 1e-6:
                bestValMacroF1 = valMetrics['macroF1']
                epochsWithoutImprovement = 0
                torch.save({
                    'experimentName': experimentName,
                    'modelName': modelName,
                    'stateDict': model.state_dict(),
                    'classNames': list(CLASS_NAMES),
                    'imageSize': imageSize,
                    'randomSeed': randomSeed,
                    'bestEpoch': epochNumber,
                    'bestValMacroF1': float(bestValMacroF1),
                    'sourceDomain': sourceDomain,
                    'externalDomain': externalDomain,
                    'coralWeight': coralWeight if experimentName == 'coral' else 0.0,
                    'targetLabelsUsed': False,
                    'trainManifestSha256': manifestSha256(trainPath),
                    'valManifestSha256': manifestSha256(valPath),
                }, checkpointPath)
            else:
                epochsWithoutImprovement += 1
            if epochsWithoutImprovement >= patience:
                stopTraining = True
                break

    elapsedSeconds = time.perf_counter() - startTime
    pd.DataFrame(historyRows).to_csv(artifactRoot / 'trainingHistory.csv', index=False)
    checkpoint = torch.load(checkpointPath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['stateDict'])
    sourcePredictions = predictFrame(model, sourceTestLoader, device)
    externalPredictions = predictFrame(model, externalTestLoader, device)
    bootstrapCount = 1000 if runMode == 'full' else 100
    sourceMetrics = calculateMetrics(sourcePredictions)
    externalMetrics = calculateMetrics(externalPredictions)
    sourceMetrics['confidenceIntervals'] = bootstrapIntervals(sourcePredictions, randomSeed, bootstrapCount)
    externalMetrics['confidenceIntervals'] = bootstrapIntervals(externalPredictions, randomSeed + 10000, bootstrapCount)
    sourceMetrics['domain'] = sourceDomain
    externalMetrics['domain'] = externalDomain
    domainShiftDelta = {
        metricName: float(externalMetrics[metricName] - sourceMetrics[metricName])
        for metricName in ('macroF1', 'balancedAccuracy')
    }
    sourcePredictions.to_csv(artifactRoot / 'sourceTestPredictions.csv', index=False)
    externalPredictions.to_csv(artifactRoot / 'externalTestPredictions.csv', index=False)
    result = {
        'experimentName': experimentName,
        'runMode': runMode,
        'paperResultsEligible': runMode == 'full',
        'randomSeed': randomSeed,
        'modelName': modelName,
        'sourceDomain': sourceDomain,
        'externalDomain': externalDomain,
        'sourceTest': sourceMetrics,
        'externalTest': externalMetrics,
        'domainShiftDelta': domainShiftDelta,
        'bestEpoch': checkpoint['bestEpoch'],
        'bestValMacroF1': checkpoint['bestValMacroF1'],
        'elapsedSeconds': elapsedSeconds,
        'coralWeight': checkpoint['coralWeight'],
        'targetLabelsUsed': checkpoint['targetLabelsUsed'],
        'selectionDataset': f'{sourceDomain}Validation',
        'externalTestUsedForSelection': False,
    }
    (artifactRoot / 'evaluationSummary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


@torch.inference_mode()
def extractDomainFeatures(projectRoot: Path, frame: pd.DataFrame, device: torch.device, batchSize: int) -> tuple[np.ndarray, pd.DataFrame]:
    _, evaluationTransform = buildTransforms(224)
    dataLoader = DataLoader(SkinDataset(projectRoot, frame, evaluationTransform), batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).to(device)
    model.eval()
    featureBlocks = []
    metadataRows = []
    for images, labels, imageIds in tqdm(dataLoader, desc='Extracting domain features'):
        images = images.to(device, non_blocking=True)
        ampContext = torch.amp.autocast('cuda', enabled=device.type == 'cuda') if device.type == 'cuda' else nullcontext()
        with ampContext:
            features = model.features(images)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
        featureBlocks.append(features.float().cpu().numpy())
        metadataRows.extend({'imageId': imageId, 'label': CLASS_NAMES[int(label)]} for imageId, label in zip(imageIds, labels))
    return np.concatenate(featureBlocks), pd.DataFrame(metadataRows)


def runDomainAudit(projectRoot: Path, runMode: str) -> dict:
    processedRoot = projectRoot / 'data' / 'processed'
    artifactRoot = projectRoot / 'artifacts' / 'extensions' / runMode / 'domainAudit'
    figureRoot = projectRoot / 'reports' / 'figures' / 'extensions' / runMode / 'domainAudit'
    artifactRoot.mkdir(parents=True, exist_ok=True)
    figureRoot.mkdir(parents=True, exist_ok=True)
    clinicalFrame = pd.concat([
        pd.read_csv(processedRoot / 'clinicalTrain.csv', low_memory=False),
        pd.read_csv(processedRoot / 'clinicalVal.csv', low_memory=False),
        pd.read_csv(processedRoot / 'clinicalTest.csv', low_memory=False),
    ], ignore_index=True)
    isicFrame = pd.read_csv(processedRoot / 'isicTest.csv', low_memory=False)
    sampleCount = 400 if runMode == 'smoke' else 2000

    def stratifiedSample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
        fractionCount = max(1, count // len(CLASS_NAMES))
        sampledGroups = [
            group.sample(n=min(len(group), fractionCount), random_state=42)
            for _, group in frame.groupby('harmonizedLabel', sort=False)
        ]
        sampled = pd.concat(sampledGroups, ignore_index=True)
        assert 'harmonizedLabel' in sampled.columns and set(sampled['harmonizedLabel']) == set(CLASS_NAMES)
        return sampled

    clinicalSample = stratifiedSample(clinicalFrame, sampleCount)
    isicSample = stratifiedSample(isicFrame, sampleCount)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batchSize = int(os.environ.get('SKIN_EVAL_BATCH_SIZE', '64' if device.type == 'cuda' else '16'))
    clinicalFeatures, clinicalMetadata = extractDomainFeatures(projectRoot, clinicalSample, device, batchSize)
    isicFeatures, isicMetadata = extractDomainFeatures(projectRoot, isicSample, device, batchSize)
    clinicalMean = clinicalFeatures.mean(axis=0)
    isicMean = isicFeatures.mean(axis=0)
    clinicalCovariance = np.cov(clinicalFeatures, rowvar=False)
    isicCovariance = np.cov(isicFeatures, rowvar=False)
    meanDistanceSquared = float(np.square(clinicalMean - isicMean).sum())
    covarianceDistance = float(np.linalg.norm(clinicalCovariance - isicCovariance, ord='fro'))

    combinedFeatures = np.vstack([clinicalFeatures, isicFeatures])
    projection = PCA(n_components=2, random_state=42).fit_transform(combinedFeatures)
    projectionFrame = pd.concat([
        clinicalMetadata.assign(domain='clinical'),
        isicMetadata.assign(domain='isic'),
    ], ignore_index=True)
    projectionFrame['principalComponent1'] = projection[:, 0]
    projectionFrame['principalComponent2'] = projection[:, 1]
    projectionFrame.to_csv(artifactRoot / 'domainFeatureProjection.csv', index=False)
    figure, axis = plt.subplots(figsize=(7, 5))
    for domainName, domainRows in projectionFrame.groupby('domain'):
        axis.scatter(domainRows['principalComponent1'], domainRows['principalComponent2'], s=10, alpha=0.45, label=domainName)
    axis.set(title='ImageNet feature distributions by dataset', xlabel='Principal component 1', ylabel='Principal component 2')
    axis.legend()
    figure.tight_layout()
    figure.savefig(figureRoot / 'domainFeaturePca.png', dpi=300, bbox_inches='tight')
    plt.close(figure)

    partitionFrames = {
        'clinicalTrain': pd.read_csv(processedRoot / 'clinicalTrain.csv', low_memory=False),
        'clinicalVal': pd.read_csv(processedRoot / 'clinicalVal.csv', low_memory=False),
        'clinicalTest': pd.read_csv(processedRoot / 'clinicalTest.csv', low_memory=False),
        'isicOfficialTrain': pd.read_csv(processedRoot / 'isicTrain.csv', low_memory=False),
        'isicOfficialTest': pd.read_csv(processedRoot / 'isicTest.csv', low_memory=False),
    }
    pathRows = []
    for partitionName, partitionFrame in partitionFrames.items():
        pathRows.extend({
            'imagePath': row.imagePath,
            'partition': partitionName,
            'label': row.harmonizedLabel,
        } for row in partitionFrame[['imagePath', 'harmonizedLabel']].itertuples(index=False))
    pathIndex = pd.DataFrame(pathRows).drop_duplicates('imagePath').set_index('imagePath')
    nearCandidates = pd.read_csv(processedRoot / 'nearDuplicateCandidates.csv', low_memory=False)
    nearCandidates['leftPartition'] = nearCandidates['leftImagePath'].map(pathIndex['partition'])
    nearCandidates['rightPartition'] = nearCandidates['rightImagePath'].map(pathIndex['partition'])
    nearCandidates['leftLabel'] = nearCandidates['leftImagePath'].map(pathIndex['label'])
    nearCandidates['rightLabel'] = nearCandidates['rightImagePath'].map(pathIndex['label'])
    crossPartition = nearCandidates.loc[
        nearCandidates['leftPartition'].notna()
        & nearCandidates['rightPartition'].notna()
        & nearCandidates['leftPartition'].ne(nearCandidates['rightPartition'])
    ].copy()
    crossPartition['sameLabel'] = crossPartition['leftLabel'].eq(crossPartition['rightLabel'])
    crossPartition['auditPriority'] = np.select(
        [crossPartition['pHashDistance'].le(2), crossPartition['pHashDistance'].le(4)],
        ['high', 'medium'],
        default='standard',
    )
    crossPartition = crossPartition.sort_values(['pHashDistance', 'leftImageId', 'rightImageId'])
    crossPartition.to_csv(artifactRoot / 'nearDuplicateCrossPartitionAudit.csv', index=False)
    scopeCounts = crossPartition.groupby(['leftPartition', 'rightPartition']).size().reset_index(name='candidateCount')
    scopeCounts.to_csv(artifactRoot / 'nearDuplicateCrossPartitionCounts.csv', index=False)

    montageRows = crossPartition.head(8)
    if len(montageRows):
        figure, axes = plt.subplots(len(montageRows), 2, figsize=(8, 3 * len(montageRows)), squeeze=False)
        for rowIndex, row in enumerate(montageRows.itertuples(index=False)):
            for columnIndex, sideName in enumerate(('left', 'right')):
                pathValue = getattr(row, f'{sideName}ImagePath')
                with Image.open(fromProjectPath(projectRoot, pathValue)) as image:
                    axes[rowIndex, columnIndex].imshow(image.convert('RGB'))
                partitionName = getattr(row, f'{sideName}Partition')
                labelName = getattr(row, f'{sideName}Label')
                axes[rowIndex, columnIndex].set_title(f'{partitionName} | {labelName} | pHash {row.pHashDistance}')
                axes[rowIndex, columnIndex].axis('off')
        figure.tight_layout()
        figure.savefig(figureRoot / 'nearDuplicatePriorityMontage.png', dpi=200, bbox_inches='tight')
        plt.close(figure)

    report = {
        'runMode': runMode,
        'featureExtractor': 'ImageNet-pretrained MobileNetV3-Small pooled features',
        'clinicalFeatureSampleCount': len(clinicalSample),
        'isicFeatureSampleCount': len(isicSample),
        'featureMeanDistanceSquared': meanDistanceSquared,
        'featureCovarianceFrobeniusDistance': covarianceDistance,
        'pcaExplainedVarianceRatio': PCA(n_components=2, random_state=42).fit(combinedFeatures).explained_variance_ratio_.tolist(),
        'nearDuplicateCandidateCount': len(nearCandidates),
        'crossPartitionNearDuplicateCandidateCount': len(crossPartition),
        'highPriorityCrossPartitionCandidateCount': int(crossPartition['auditPriority'].eq('high').sum()),
        'nearDuplicateStatus': 'Candidates require visual review and are not treated as confirmed duplicates.',
    }
    (artifactRoot / 'domainAuditSummary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def aggregateExtensionResults(projectRoot: Path, runMode: str, randomSeeds: tuple[int, ...]) -> dict:
    experimentNames = ('efficientnet', 'reverse', 'coral')
    resultRows = []
    perSeedResults = {}
    baselineResults = {}
    for experimentName in experimentNames:
        perSeedResults[experimentName] = {}
        for randomSeed in randomSeeds:
            resultPath = projectRoot / 'artifacts' / 'extensions' / runMode / experimentName / f'seed{randomSeed}' / 'evaluationSummary.json'
            if not resultPath.exists():
                raise FileNotFoundError(f'Missing {resultPath}.')
            result = json.loads(resultPath.read_text(encoding='utf-8'))
            if result['randomSeed'] != randomSeed or result['experimentName'] != experimentName:
                raise ValueError(f'Artifact metadata mismatch: {resultPath}')
            if result['externalTestUsedForSelection']:
                raise ValueError(f'External-test selection leakage: {resultPath}')
            if experimentName == 'coral' and result['targetLabelsUsed']:
                raise ValueError(f'CORAL target labels were used: {resultPath}')
            perSeedResults[experimentName][str(randomSeed)] = result
            for testName in ('sourceTest', 'externalTest'):
                resultRows.append({
                    'experiment': experimentName,
                    'randomSeed': randomSeed,
                    'testRole': testName,
                    'domain': result[testName]['domain'],
                    'macroF1': result[testName]['macroF1'],
                    'balancedAccuracy': result[testName]['balancedAccuracy'],
                })
    for randomSeed in randomSeeds:
        baselinePath = projectRoot / 'artifacts' / 'ds' / runMode / f'seed{randomSeed}' / 'evaluationSummary.json'
        if not baselinePath.exists():
            raise FileNotFoundError(f'Missing preserved baseline artifact: {baselinePath}')
        baselineResult = json.loads(baselinePath.read_text(encoding='utf-8'))
        if baselineResult.get('randomSeed') != randomSeed:
            raise ValueError(f'Baseline seed mismatch: {baselinePath}')
        baselineResults[str(randomSeed)] = baselineResult
    resultFrame = pd.DataFrame(resultRows)
    aggregateRows = []
    for keys, group in resultFrame.groupby(['experiment', 'testRole', 'domain'], sort=False):
        experimentName, testRole, domainName = keys
        for metricName in ('macroF1', 'balancedAccuracy'):
            aggregateRows.append({
                'experiment': experimentName,
                'testRole': testRole,
                'domain': domainName,
                'metric': metricName,
                'mean': float(group[metricName].mean()),
                'standardDeviation': float(group[metricName].std(ddof=1)) if len(group) > 1 else 0.0,
            })
    artifactRoot = projectRoot / 'artifacts' / 'extensions' / runMode
    artifactRoot.mkdir(parents=True, exist_ok=True)
    resultFrame.to_csv(artifactRoot / 'extensionPerSeedMetrics.csv', index=False)
    pd.DataFrame(aggregateRows).to_csv(artifactRoot / 'extensionAggregateMetrics.csv', index=False)
    comparisonRows = []
    for randomSeed in randomSeeds:
        baselineResult = baselineResults[str(randomSeed)]
        for experimentName in ('efficientnet', 'coral'):
            extensionResult = perSeedResults[experimentName][str(randomSeed)]
            for metricName in ('macroF1', 'balancedAccuracy'):
                comparisonRows.append({
                    'experiment': experimentName,
                    'randomSeed': randomSeed,
                    'metric': metricName,
                    'externalScoreDifferenceFromBaseline': float(
                        extensionResult['externalTest'][metricName]
                        - baselineResult['isicDomainShiftTest'][metricName]
                    ),
                    'gapChangeFromBaseline': float(
                        extensionResult['domainShiftDelta'][metricName]
                        - baselineResult['absoluteDomainShiftDelta'][metricName]
                    ),
                })
    comparisonFrame = pd.DataFrame(comparisonRows)
    comparisonFrame.to_csv(artifactRoot / 'extensionBaselineComparisons.csv', index=False)
    aggregateComparisons = []
    for keys, group in comparisonFrame.groupby(['experiment', 'metric'], sort=False):
        experimentName, metricName = keys
        for comparisonName in ('externalScoreDifferenceFromBaseline', 'gapChangeFromBaseline'):
            aggregateComparisons.append({
                'experiment': experimentName,
                'metric': metricName,
                'comparison': comparisonName,
                'mean': float(group[comparisonName].mean()),
                'standardDeviation': float(group[comparisonName].std(ddof=1)) if len(group) > 1 else 0.0,
            })
    summary = {
        'runMode': runMode,
        'paperResultsEligible': runMode == 'full',
        'randomSeeds': list(randomSeeds),
        'aggregateMetrics': aggregateRows,
        'aggregateBaselineComparisons': aggregateComparisons,
        'comparisonInterpretation': 'Positive external-score difference is better than baseline; positive gap change means a less negative cross-dataset gap.',
        'baselineResults': baselineResults,
        'perSeedResults': perSeedResults,
        'interpretationBoundary': 'Cross-dataset gaps are consistent with domain shift but do not isolate acquisition modality as a causal effect.',
    }
    (artifactRoot / 'extensionSummary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
