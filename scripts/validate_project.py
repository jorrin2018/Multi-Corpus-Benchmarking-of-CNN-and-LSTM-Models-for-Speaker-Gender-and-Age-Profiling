#!/usr/bin/env python3
"""
Script de validación completa del proyecto Speaker Profiling.

Este script ejecuta todas las validaciones de calidad, tests y verificaciones
para asegurar que el proyecto cumple con los estándares científicos y técnicos.
"""

import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import json
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectValidator:
    """Validador completo del proyecto."""
    
    def __init__(self, project_root: str = "."):
        """Inicializar validador."""
        self.project_root = Path(project_root)
        self.results = {}
        self.start_time = time.time()
    
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Ejecutar comando y capturar resultado."""
        logger.info(f"Ejecutando: {description}")
        logger.debug(f"Comando: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minutos timeout
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'description': description
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timeout',
                'description': description
            }
        except Exception as e:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'description': description
            }
    
    def validate_code_quality(self) -> Dict[str, Any]:
        """Validar calidad de código."""
        logger.info("🔍 Validando calidad de código...")
        
        quality_checks = {
            'flake8': self.run_command(
                ['python', '-m', 'flake8', 'src'],
                'Linting con flake8'
            ),
            'mypy': self.run_command(
                ['python', '-m', 'mypy', 'src', '--ignore-missing-imports'],
                'Type checking con mypy'
            ),
            'black_check': self.run_command(
                ['python', '-m', 'black', '--check', 'src'],
                'Verificar formato con black'
            ),
            'isort_check': self.run_command(
                ['python', '-m', 'isort', '--check-only', 'src'],
                'Verificar imports con isort'
            )
        }
        
        passed = sum(1 for result in quality_checks.values() if result['success'])
        total = len(quality_checks)
        
        return {
            'passed': passed,
            'total': total,
            'success_rate': passed / total,
            'details': quality_checks
        }
    
    def validate_tests(self) -> Dict[str, Any]:
        """Ejecutar suite de tests."""
        logger.info("🧪 Ejecutando tests...")
        
        test_suites = {
            'unit_tests': self.run_command(
                ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short', 
                 '--cov=src', '--cov-report=term-missing'],
                'Tests unitarios con cobertura'
            ),
            'integration_tests': self.run_command(
                ['python', '-m', 'pytest', 'tests/integration/', '-v'],
                'Tests de integración'
            ),
            'performance_tests': self.run_command(
                ['python', '-m', 'pytest', 'tests/performance/', '-v', '--benchmark-skip'],
                'Tests de rendimiento'
            )
        }
        
        passed = sum(1 for result in test_suites.values() if result['success'])
        total = len(test_suites)
        
        return {
            'passed': passed,
            'total': total,
            'success_rate': passed / total,
            'details': test_suites
        }
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validar documentación."""
        logger.info("📚 Validando documentación...")
        
        doc_files = [
            'README.md',
            'docs/quick_start.md',
            'docs/installation.md',
            'docs/tutorials.md',
            'docs/api/README.md'
        ]
        
        missing_docs = []
        for doc_file in doc_files:
            if not (self.project_root / doc_file).exists():
                missing_docs.append(doc_file)
        
        # Verificar docstrings en módulos
        docstring_check = self.run_command(
            ['python', '-c', '''
import ast
import sys
from pathlib import Path

missing_docs = []
for py_file in Path("src").rglob("*.py"):
    if py_file.name == "__init__.py":
        continue
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            if not ast.get_docstring(tree):
                missing_docs.append(str(py_file))
    except:
        pass

if missing_docs:
    print("Files without docstrings:")
    for file in missing_docs:
        print(f"  - {file}")
    sys.exit(1)
else:
    print("All modules have documentation")
            '''],
            'Verificar docstrings en módulos'
        )
        
        return {
            'missing_files': missing_docs,
            'docstring_check': docstring_check['success'],
            'success': len(missing_docs) == 0 and docstring_check['success'],
            'details': {
                'missing_documentation_files': missing_docs,
                'docstring_validation': docstring_check
            }
        }
    
    def validate_project_structure(self) -> Dict[str, Any]:
        """Validar estructura del proyecto."""
        logger.info("📁 Validando estructura del proyecto...")
        
        required_dirs = [
            'src',
            'tests',
            'scripts',
            'docs',
            'config',
            'notebooks',
            'docker'
        ]
        
        required_files = [
            'README.md',
            'setup.py',
            'requirements.txt',
            'requirements-dev.txt',
            'LICENSE',
            '.gitignore',
            'pytest.ini',
            'mypy.ini',
            '.flake8'
        ]
        
        missing_dirs = []
        missing_files = []
        
        for directory in required_dirs:
            if not (self.project_root / directory).exists():
                missing_dirs.append(directory)
        
        for file in required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
        
        return {
            'missing_directories': missing_dirs,
            'missing_files': missing_files,
            'success': len(missing_dirs) == 0 and len(missing_files) == 0,
            'structure_complete': len(missing_dirs) == 0 and len(missing_files) == 0
        }
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validar dependencias."""
        logger.info("📦 Validando dependencias...")
        
        dependency_checks = {
            'pip_check': self.run_command(
                ['python', '-m', 'pip', 'check'],
                'Verificar consistencia de dependencias'
            ),
            'safety_check': self.run_command(
                ['python', '-m', 'safety', 'check'],
                'Verificar vulnerabilidades de seguridad'
            ),
            'requirements_install': self.run_command(
                ['python', '-m', 'pip', 'install', '-r', 'requirements.txt', '--dry-run'],
                'Verificar instalación de requirements'
            )
        }
        
        passed = sum(1 for result in dependency_checks.values() if result['success'])
        total = len(dependency_checks)
        
        return {
            'passed': passed,
            'total': total,
            'success_rate': passed / total,
            'details': dependency_checks
        }
    
    def validate_scripts(self) -> Dict[str, Any]:
        """Validar scripts principales."""
        logger.info("⚙️ Validando scripts...")
        
        scripts = [
            'scripts/train.py',
            'scripts/evaluate.py',
            'scripts/predict.py',
            'scripts/benchmark.py',
            'scripts/reproduce_paper.py'
        ]
        
        script_checks = {}
        for script in scripts:
            script_name = Path(script).stem
            script_checks[script_name] = self.run_command(
                ['python', script, '--help'],
                f'Verificar script {script_name}'
            )
        
        passed = sum(1 for result in script_checks.values() if result['success'])
        total = len(script_checks)
        
        return {
            'passed': passed,
            'total': total,
            'success_rate': passed / total,
            'details': script_checks
        }
    
    def validate_scientific_reproducibility(self) -> Dict[str, Any]:
        """Validar reproducibilidad científica."""
        logger.info("🔬 Validando reproducibilidad científica...")
        
        reproducibility_checks = {
            'config_validation': self.run_command(
                ['python', '-c', '''
import yaml
from pathlib import Path

configs = ["config/datasets.yaml", "config/models.yaml", "config/training.yaml"]
for config_file in configs:
    if not Path(config_file).exists():
        print(f"Missing config: {config_file}")
        exit(1)
    try:
        with open(config_file) as f:
            yaml.safe_load(f)
        print(f"Valid config: {config_file}")
    except Exception as e:
        print(f"Invalid config {config_file}: {e}")
        exit(1)
print("All configs are valid")
                '''],
                'Validar archivos de configuración'
            ),
            'import_test': self.run_command(
                ['python', '-c', '''
try:
    from src.models.cnn_models import CNNModelFactory
    from src.models.lstm_models import LSTMModelFactory
    from src.datasets.voxceleb1 import VoxCeleb1Dataset
    from src.training.trainer import SpeakerProfilingTrainer
    from src.evaluation.metrics import MetricsCalculator
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)
                '''],
                'Verificar imports principales'
            )
        }
        
        passed = sum(1 for result in reproducibility_checks.values() if result['success'])
        total = len(reproducibility_checks)
        
        return {
            'passed': passed,
            'total': total,
            'success_rate': passed / total,
            'details': reproducibility_checks
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generar reporte resumen."""
        elapsed_time = time.time() - self.start_time
        
        # Calcular estadísticas generales
        total_checks = 0
        passed_checks = 0
        
        for category, results in self.results.items():
            if isinstance(results, dict) and 'passed' in results and 'total' in results:
                total_checks += results['total']
                passed_checks += results['passed']
        
        overall_success_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        # Determinar estado general
        if overall_success_rate >= 0.95:
            status = "EXCELLENT"
            status_emoji = "🎉"
        elif overall_success_rate >= 0.85:
            status = "GOOD"
            status_emoji = "✅"
        elif overall_success_rate >= 0.70:
            status = "ACCEPTABLE"
            status_emoji = "⚠️"
        else:
            status = "NEEDS_IMPROVEMENT"
            status_emoji = "❌"
        
        return {
            'overall_status': status,
            'status_emoji': status_emoji,
            'success_rate': overall_success_rate,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'execution_time': elapsed_time,
            'categories': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Ejecutar validación completa."""
        logger.info("🚀 Iniciando validación completa del proyecto...")
        
        # Ejecutar todas las validaciones
        self.results['project_structure'] = self.validate_project_structure()
        self.results['dependencies'] = self.validate_dependencies()
        self.results['code_quality'] = self.validate_code_quality()
        self.results['tests'] = self.validate_tests()
        self.results['documentation'] = self.validate_documentation()
        self.results['scripts'] = self.validate_scripts()
        self.results['scientific_reproducibility'] = self.validate_scientific_reproducibility()
        
        # Generar reporte final
        summary = self.generate_summary_report()
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Imprimir resumen de validación."""
        print("\n" + "="*80)
        print(f"  {summary['status_emoji']} REPORTE DE VALIDACIÓN DEL PROYECTO")
        print("="*80)
        print(f"Estado General: {summary['overall_status']}")
        print(f"Tasa de Éxito: {summary['success_rate']:.1%}")
        print(f"Checks Pasados: {summary['passed_checks']}/{summary['total_checks']}")
        print(f"Tiempo de Ejecución: {summary['execution_time']:.1f}s")
        print(f"Fecha: {summary['timestamp']}")
        
        print("\n📊 RESULTADOS POR CATEGORÍA:")
        print("-" * 50)
        
        for category, results in summary['categories'].items():
            if isinstance(results, dict):
                if 'success_rate' in results:
                    rate = results['success_rate']
                    emoji = "✅" if rate >= 0.8 else "⚠️" if rate >= 0.6 else "❌"
                    print(f"{emoji} {category.replace('_', ' ').title()}: {rate:.1%} ({results['passed']}/{results['total']})")
                elif 'success' in results:
                    emoji = "✅" if results['success'] else "❌"
                    print(f"{emoji} {category.replace('_', ' ').title()}: {'PASS' if results['success'] else 'FAIL'}")
        
        if summary['failed_checks'] > 0:
            print(f"\n⚠️ Se encontraron {summary['failed_checks']} problemas.")
            print("Revisar los logs detallados para más información.")
        else:
            print("\n🎉 ¡Todas las validaciones pasaron exitosamente!")
        
        print("="*80 + "\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Validar calidad completa del proyecto")
    parser.add_argument('--project-root', default='.', help='Directorio raíz del proyecto')
    parser.add_argument('--output', help='Archivo de salida para el reporte JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Output verbose')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ejecutar validación
    validator = ProjectValidator(args.project_root)
    summary = validator.run_full_validation()
    
    # Mostrar resumen
    validator.print_summary(summary)
    
    # Guardar reporte si se especifica
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Reporte guardado en: {args.output}")
    
    # Salir con código de error si hay problemas
    if summary['success_rate'] < 0.8:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main() 