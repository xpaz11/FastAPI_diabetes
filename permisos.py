from sqlalchemy import Column, Integer, String, ForeignKey, Table, create_engine, text
from sqlalchemy.orm import relationship, declarative_base
import os

Base = declarative_base()

# Tabla intermedia para relación muchos a muchos entre usuarios y roles
usuario_rol = Table(
    'usuario_rol', Base.metadata,
    Column('usuario_id', Integer, ForeignKey('usuarios.id')),
    Column('rol_id', Integer, ForeignKey('roles.id'))
)

# Tabla intermedia para relación muchos a muchos entre usuarios y grupos
usuario_grupo = Table(
    'usuario_grupo', Base.metadata,
    Column('usuario_id', Integer, ForeignKey('usuarios.id')),
    Column('grupo_id', Integer, ForeignKey('grupos.id'))
)

class Usuario(Base):
    __tablename__ = 'usuarios'
    id = Column(Integer, primary_key=True)
    nombre = Column(String)
    email = Column(String, unique=True)
    roles = relationship('Rol', secondary=usuario_rol, back_populates='usuarios')
    grupos = relationship('Grupo', secondary=usuario_grupo, back_populates='usuarios')

class Rol(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True)
    nombre = Column(String)
    usuarios = relationship('Usuario', secondary=usuario_rol, back_populates='roles')

class Grupo(Base):
    __tablename__ = 'grupos'
    id = Column(Integer, primary_key=True)
    nombre = Column(String)
    usuarios = relationship('Usuario', secondary=usuario_grupo, back_populates='grupos')

# ✅ Conexión a la base de datos Neon
DATABASE_URL = os.getenv("DATABASE_URL", 'postgresql://neondb_owner:npg_BDG2IiT0aqAy@ep-super-heart-agp17yzq-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require')
engine = create_engine(DATABASE_URL)

# ✅ Funciones para permisos
def obtener_roles(usuario):
    """Devuelve los roles asociados a un usuario"""
    query = text("""
        SELECT r.nombre FROM roles r
        JOIN usuario_rol ur ON r.id = ur.rol_id
        JOIN usuarios u ON u.id = ur.usuario_id
        WHERE u.nombre = :usuario
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"usuario": usuario}).fetchall()
    return [row[0] for row in result]

def tiene_permiso(usuario, permiso_requerido):
    """Verifica si el usuario tiene un rol específico"""
    roles = obtener_roles(usuario)
    return permiso_requerido in roles
