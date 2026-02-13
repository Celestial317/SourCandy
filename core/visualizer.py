from typing import Optional, List
import numpy as np
from schema import SournessMap, TokenDiagnosis


class SournessVisualizer:
    '''
    Visualization utilities for sourness maps and diagnostic data.
    Generates heatmaps and plots for visual debugging of token-level diagnostics.
    Compatible with matplotlib and seaborn for rendering.
    '''
    
    def __init__(self):
        '''
        Initialize visualizer. Rendering backends are imported on-demand.
        '''
        self.matplotlibAvailable = False
        self.seabornAvailable = False
        
        try:
            import matplotlib
            self.matplotlibAvailable = True
        except ImportError:
            pass
            
        try:
            import seaborn
            self.seabornAvailable = True
        except ImportError:
            pass
            
    def plotSournessHeatmap(
        self,
        sournessMap: SournessMap,
        title: str = 'Token Sourness Heatmap',
        savePath: Optional[str] = None,
        showPlot: bool = True
    ) -> Optional[str]:
        '''
        Generate heatmap visualization of token sourness scores.
        Maps tokens to their sourness values with color-coded intensity.
        
        Args:
            sournessMap: SournessMap containing token diagnostics
            title: Plot title
            savePath: Optional path to save figure
            showPlot: Whether to display plot interactively
            
        Returns:
            Path to saved figure or None if not saved
        '''
        if not self.matplotlibAvailable:
            return self.generateTextHeatmap(sournessMap)
            
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        diagnostics = sournessMap.tokenDiagnostics
        
        if not diagnostics:
            print('No tokens to visualize')
            return None
            
        tokens = [d.token for d in diagnostics]
        scores = [d.sournessScore for d in diagnostics]
        
        tokensPerRow = 10
        numRows = (len(tokens) + tokensPerRow - 1) // tokensPerRow
        
        heatmapData = np.zeros((numRows, tokensPerRow))
        tokenGrid = [['' for _ in range(tokensPerRow)] for _ in range(numRows)]
        
        for i, (token, score) in enumerate(zip(tokens, scores)):
            row = i // tokensPerRow
            col = i % tokensPerRow
            heatmapData[row, col] = score
            tokenGrid[row][col] = token[:8]
            
        fig, ax = plt.subplots(figsize=(15, max(3, numRows)))
        
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'sourness',
            ['#2ECC71', '#F39C12', '#E74C3C']
        )
        
        im = ax.imshow(heatmapData, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(tokensPerRow))
        ax.set_yticks(np.arange(numRows))
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        for i in range(numRows):
            for j in range(tokensPerRow):
                if tokenGrid[i][j]:
                    text = ax.text(
                        j, i, tokenGrid[i][j],
                        ha='center', va='center',
                        color='white' if heatmapData[i, j] > 0.5 else 'black',
                        fontsize=8,
                        weight='bold'
                    )
                    
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sourness Score', rotation=270, labelpad=20)
        
        ax.set_title(f'{title}\nAggregate Score: {sournessMap.aggregateSournessScore:.3f}')
        
        plt.tight_layout()
        
        if savePath:
            plt.savefig(savePath, dpi=150, bbox_inches='tight')
            print(f'Heatmap saved to {savePath}')
            
        if showPlot:
            plt.show()
        else:
            plt.close()
            
        return savePath if savePath else None
    
    def generateTextHeatmap(self, sournessMap: SournessMap) -> str:
        '''
        Generate ASCII text-based heatmap when matplotlib unavailable.
        
        Args:
            sournessMap: SournessMap to visualize
            
        Returns:
            Formatted string representation of heatmap
        '''
        diagnostics = sournessMap.tokenDiagnostics
        
        if not diagnostics:
            return 'No tokens to visualize'
            
        output = ['\n' + '='*70]
        output.append('TOKEN SOURNESS HEATMAP (Text Mode)')
        output.append(f'Aggregate Score: {sournessMap.aggregateSournessScore:.3f}')
        output.append('='*70 + '\n')
        
        for i, diag in enumerate(diagnostics):
            barLength = 40
            filledLength = int(barLength * diag.sournessScore)
            
            bar = '█' * filledLength + '░' * (barLength - filledLength)
            
            colorCode = self.getColorCode(diag.sournessScore)
            
            output.append(
                f'{i+1:3d}. {diag.token:15s} [{bar}] {diag.sournessScore:.3f} {colorCode}'
            )
            
        output.append('\n' + '='*70)
        
        return '\n'.join(output)
    
    def getColorCode(self, score: float) -> str:
        '''
        Get text indicator for sourness level.
        
        Args:
            score: Sourness score
            
        Returns:
            Visual indicator string
        '''
        if score < 0.3:
            return '🟢 Sweet'
        elif score < 0.6:
            return '🟡 Mild'
        elif score < 0.8:
            return '🟠 Sour'
        else:
            return '🔴 Very Sour'
            
    def plotSournessDistribution(
        self,
        sournessMap: SournessMap,
        savePath: Optional[str] = None,
        showPlot: bool = True
    ) -> Optional[str]:
        '''
        Plot distribution histogram of sourness scores.
        
        Args:
            sournessMap: SournessMap to analyze
            savePath: Optional save path
            showPlot: Whether to display plot
            
        Returns:
            Save path or None
        '''
        if not self.matplotlibAvailable:
            return self.generateTextDistribution(sournessMap)
            
        import matplotlib.pyplot as plt
        
        scores = [d.sournessScore for d in sournessMap.tokenDiagnostics]
        
        if not scores:
            print('No scores to plot')
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(scores, bins=20, color='#3498DB', edgecolor='black', alpha=0.7)
        ax.axvline(
            sournessMap.aggregateSournessScore,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Aggregate: {sournessMap.aggregateSournessScore:.3f}'
        )
        
        ax.set_xlabel('Sourness Score')
        ax.set_ylabel('Token Count')
        ax.set_title('Distribution of Token Sourness Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if savePath:
            plt.savefig(savePath, dpi=150, bbox_inches='tight')
            
        if showPlot:
            plt.show()
        else:
            plt.close()
            
        return savePath if savePath else None
    
    def generateTextDistribution(self, sournessMap: SournessMap) -> str:
        '''
        Generate text-based distribution summary.
        
        Args:
            sournessMap: SournessMap to analyze
            
        Returns:
            Formatted distribution summary
        '''
        scores = [d.sournessScore for d in sournessMap.tokenDiagnostics]
        
        if not scores:
            return 'No scores to analyze'
            
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        counts = [0] * (len(bins) - 1)
        
        for score in scores:
            for i in range(len(bins) - 1):
                if bins[i] <= score < bins[i+1]:
                    counts[i] += 1
                    break
            else:
                if score == 1.0:
                    counts[-1] += 1
                    
        output = ['\n' + '='*60]
        output.append('SOURNESS SCORE DISTRIBUTION')
        output.append('='*60 + '\n')
        
        maxCount = max(counts) if counts else 1
        
        for label, count in zip(labels, counts):
            barLength = int((count / maxCount) * 40) if maxCount > 0 else 0
            bar = '█' * barLength
            percentage = (count / len(scores)) * 100 if scores else 0
            output.append(f'{label}: {bar} {count} ({percentage:.1f}%)')
            
        output.append(f'\nTotal Tokens: {len(scores)}')
        output.append(f'Aggregate Score: {sournessMap.aggregateSournessScore:.3f}')
        output.append('='*60)
        
        return '\n'.join(output)


def plotSournessHeatmap(
    sournessMap: SournessMap,
    title: str = 'Token Sourness Heatmap',
    savePath: Optional[str] = None,
    showPlot: bool = True
) -> Optional[str]:
    '''
    Convenience function to plot sourness heatmap.
    
    Args:
        sournessMap: SournessMap to visualize
        title: Plot title
        savePath: Optional save location
        showPlot: Whether to display interactively
        
    Returns:
        Path to saved figure or None
    '''
    visualizer = SournessVisualizer()
    return visualizer.plotSournessHeatmap(sournessMap, title, savePath, showPlot)
