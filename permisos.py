from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import relationship, declarative_base

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