#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
W ve L Pattern Analiz Uygulaması
Geliştirilmiş Hibrit Analiz Modeli
"""

import numpy as np
from models.base_model import BaseAnalysisModel

class HibritAnalysis(BaseAnalysisModel):
    """Geliştirilmiş hibrit model - Dinamik ağırlıklandırma, durum tespiti ve kalibrasyon"""
    
    def __init__(self):
        super().__init__()
        self.name = "Hibrit Analiz"
        self.description = "Tüm modelleri akıllı birleştirme algoritması ile değerlendiren gelişmiş hibrit model."
        self.min_data_points = 5
        
        # Minimum tahmin sayısı
        self.min_predictions = 3
        
        # Durum tespiti için geçmiş durum ağırlıkları
        self.pattern_state_weights = {
            "consecutive_wins": {"T-Şekli": 1.2, "Serpme": 1.1, "Sınır": 1.3},
            "consecutive_losses": {"L-Şekli": 1.2, "Komşuluk": 1.1, "Isı Haritası": 1.3},
            "alternating": {"Çapraz (Diagonal)": 1.3, "Zig-Zag": 1.4, "Simetri": 1.2},
            "win_dominant": {"Dikdörtgen": 1.2, "Spiral": 1.1, "Kuadran": 1.0},
            "loss_dominant": {"Isı Haritası": 1.2, "Sınır": 1.1, "Serpme": 1.0},
            "balanced": {"Karma Analiz": 1.1, "Kuadran": 1.0, "Komşuluk": 1.0}
        }
    
    def analyze(self, matrix, history=None, model_stats=None):
        """
        Hibrit analiz yapar
        
        Args:
            matrix (numpy.ndarray): 5x5 matris, 0=boş, 1=W, 2=L
            history (list, optional): Geçmiş hamlelerin listesi (row, col, value)
            model_stats (dict, optional): Mevcut model istatistikleri (UI tarafından sağlanır)
            
        Returns:
            dict: Sonuç ve güven düzeyi içeren sözlük
        """
        # Temel istatistikler
        stats = self._calculate_basic_stats(matrix)
        
        if stats['total'] < self.min_data_points:
            return {'prediction': 0, 'confidence': 0}  # Yetersiz veri
        
        # Model istatistikleri sağlanmadıysa, varsayılan tahmine dön
        if not model_stats:
            return {'prediction': stats['prediction'], 'confidence': 0.5}
        
        # Mevcut durum tespiti
        current_state = self._detect_pattern_state(history, matrix)
        
        # Modelleri filtrele - Hibrit ve Karma modeli dahil etme
        filtered_models = {k: v for k, v in model_stats.items() if k != "Hibrit Analiz" and k != "Karma Analiz"}
        
        # En az minimum tahmin sayısı kadar tahmin yapmış modelleri seç
        qualified_models = {k: v for k, v in filtered_models.items() if v["total"] >= self.min_predictions}
        
        if not qualified_models:
            # Yeterli tahmin yapan model yoksa, genel istatistiklere göre tahmin yap
            return {'prediction': stats['prediction'], 'confidence': 0.5}
        
        # Model ağırlıklarını hesapla - başarı oranına, durum tipine ve tahmin sayısına göre
        model_weights = self._calculate_weights(qualified_models, current_state)
        
        # W ve L için ağırlıklı oylar
        w_vote = 0
        l_vote = 0
        
        # Her modelden tahmin almak için tüm modelleri yeniden oluştur ve çalıştır
        from models.diagonal import DiagonalAnalysis
        from models.rectangle import RectangleAnalysis
        from models.lshape import LShapeAnalysis
        from models.tshape import TShapeAnalysis
        from models.spiral import SpiralAnalysis
        from models.neighborhood import NeighborhoodAnalysis
        from models.zigzag import ZigzagAnalysis
        from models.scatter import ScatterAnalysis
        from models.quadrant import QuadrantAnalysis
        from models.symmetry import SymmetryAnalysis
        from models.border import BorderAnalysis
        from models.heatmap import HeatmapAnalysis
        from models.combined import CombinedAnalysis
        
        models = {
            "Çapraz (Diagonal)": DiagonalAnalysis(),
            "Dikdörtgen": RectangleAnalysis(),
            "L-Şekli": LShapeAnalysis(),
            "T-Şekli": TShapeAnalysis(),
            "Spiral": SpiralAnalysis(),
            "Komşuluk": NeighborhoodAnalysis(),
            "Zig-Zag": ZigzagAnalysis(),
            "Serpme": ScatterAnalysis(),
            "Kuadran": QuadrantAnalysis(),
            "Simetri": SymmetryAnalysis(),
            "Sınır": BorderAnalysis(),
            "Isı Haritası": HeatmapAnalysis(),
            "Karma Analiz": CombinedAnalysis()
        }
        
        # Her modelin tahminini al ve ağırlıklandır
        for model_name, weight in model_weights.items():
            model = models.get(model_name)
            if model is None:
                continue
                
            prediction = model.analyze(matrix, history)
            
            if prediction == 1:  # W tahmini
                w_vote += weight
            elif prediction == 2:  # L tahmini
                l_vote += weight
        
        # Nihai tahmin ve güven değeri
        if w_vote > l_vote:
            total_vote = w_vote + l_vote
            confidence = self._calibrate_confidence(w_vote / total_vote if total_vote > 0 else 0.6)
            return {'prediction': 1, 'confidence': confidence}
        elif l_vote > w_vote:
            total_vote = w_vote + l_vote
            confidence = self._calibrate_confidence(l_vote / total_vote if total_vote > 0 else 0.6)
            return {'prediction': 2, 'confidence': confidence}
        else:
            # Eşitlik durumunda, genel istatistiklere göre karar ver
            if stats['w_probability'] > stats['l_probability']:
                return {'prediction': 1, 'confidence': 0.55}
            else:
                return {'prediction': 2, 'confidence': 0.55}
    
    def _detect_pattern_state(self, history, matrix):
        """Mevcut durum tipini tespit eder"""
        if not history or len(history) < 3:
            return "balanced"
            
        # Son 3 sonuca bak
        last_three = [h[2] for h in history[-3:]]
        
        if all(v == 1 for v in last_three):
            return "consecutive_wins"
        elif all(v == 2 for v in last_three):
            return "consecutive_losses"
        elif last_three == [1, 2, 1] or last_three == [2, 1, 2]:
            return "alternating"
        
        # W/L oranına bak
        w_count = sum(1 for _, _, v in history if v == 1)
        l_count = sum(1 for _, _, v in history if v == 2)
        
        if w_count > l_count * 1.5:
            return "win_dominant"
        elif l_count > w_count * 1.5:
            return "loss_dominant"
            
        return "balanced"
    
    def _calculate_weights(self, model_stats, current_state):
        """Model ağırlıklarını hesaplar"""
        weights = {}
        
        for model_name, stats in model_stats.items():
            # Başarı oranı ağırlığı
            success_weight = stats['success_rate'] / 100.0
            
            # Tecrübe ağırlığı - daha fazla tahmin yapmış modeller bonus alır
            experience_factor = min(stats['total'] / 20.0, 1.0) * 0.2 + 0.8
            
            # Durum temelli ağırlık - belirli durumlarda daha iyi çalışan modeller bonus alır
            state_factor = self.pattern_state_weights.get(current_state, {}).get(model_name, 1.0)
            
            # Toplam ağırlık
            weights[model_name] = success_weight * experience_factor * state_factor
        
        return weights
    
    def _calibrate_confidence(self, raw_confidence):
        """Güven değerini kalibre eder - aşırı güveni azaltır"""
        # Orta değere doğru hafif çekme (regresyon)
        calibrated = 0.5 + (raw_confidence - 0.5) * 0.8
        
        # Sınırlara yaklaşmayı zorlaştır
        if calibrated > 0.8:
            calibrated = 0.8 + (calibrated - 0.8) * 0.5
        elif calibrated < 0.2:
            calibrated = 0.2 - (0.2 - calibrated) * 0.5
            
        return calibrated
